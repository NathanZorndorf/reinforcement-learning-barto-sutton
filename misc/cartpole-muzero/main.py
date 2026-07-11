"""
Minimal MuZero scaffold for CartPole-v1.
=========================================

This is a LEARNING SKELETON, not a black box. Everything that requires PyTorch
plumbing (the three networks, the MCTS scaffolding, self-play, the replay buffer,
the training unroll + loss) is written for you. Your job is to fill in the THREE
functions marked  `# TODO(you)`  -- they are exactly the three formulas we drilled:

    1. ucb_score()            -> the PUCT selection score
    2. backpropagate()        -> the MCTS backup (fold reward + discount, update N/W)
    3. compute_value_target() -> the n-step return

None of the three requires neural-network knowledge -- they are direct
translations of the equations we covered. Once they're filled in, run this file
and watch the average episode length climb toward 500 (CartPole is "solved"
around 475+).

Requirements:  pip install gymnasium torch numpy

Simplifications vs. the real MuZero paper (fine for CartPole, revisit for Atari):
  * value & reward are predicted as raw scalars with MSE loss.  The paper uses a
    CATEGORICAL representation over a value "support" + cross-entropy (the gotcha
    we discussed). CartPole's return range is small, so scalars are fine here.
  * no observation stacking (a single CartPole obs is ~Markov).
  * MLP networks instead of ResNets.
"""

import math
import random
from dataclasses import dataclass, field
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import gymnasium as gym
except ImportError:  # allow the file to be imported / syntax-checked without gym
    gym = None


# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
@dataclass
class Config:
    env_id: str = "CartPole-v1"
    obs_dim: int = 4
    num_actions: int = 2
    hidden_dim: int = 64            # size of the latent state s^k

    num_simulations: int = 25       # MCTS simulations per real move
    discount: float = 0.997         # gamma
    num_unroll_steps: int = 5       # K   (how far we unroll the model in training)
    td_steps: int = 10              # n   (horizon of the n-step return)

    # PUCT exploration constants (MuZero form). If you use the simpler AlphaZero
    # form, you only need a single constant ~1.25 and can ignore c2.
    c1: float = 1.25
    c2: float = 19652.0

    dirichlet_alpha: float = 0.3
    root_exploration_frac: float = 0.25

    batch_size: int = 128
    replay_capacity: int = 500      # number of games kept
    lr: float = 2e-3

    num_iterations: int = 200       # outer loop: (self-play games -> train)
    games_per_iteration: int = 5
    train_steps_per_iteration: int = 20

    epsilon = 1e-3
    
    support_min = -300
    support_max = 300
    num_support_bins = abs(support_min) + abs(support_max) + 1  # support bins for value/reward distribution
    


# ----------------------------------------------------------------------------
# Networks:  h (representation), g (dynamics), f (prediction)
# ----------------------------------------------------------------------------
class MuZeroNet(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        h = cfg.hidden_dim
        self.num_actions = cfg.num_actions

        # h_theta : raw observation  ->  latent state  s^0
        self.representation = nn.Sequential(
            nn.Linear(cfg.obs_dim, h), nn.ReLU(),
            nn.Linear(h, h), nn.ReLU(),
        )

        # g_theta : (latent state, action)  ->  (next latent state, reward)
        # the action is fed in as a one-hot vector concatenated to the state
        # (for image latents you'd instead broadcast it as extra channel planes).
        self.dynamics = nn.Sequential(
            nn.Linear(h + cfg.num_actions, h), nn.ReLU(),
            nn.Linear(h, h), nn.ReLU(),
        )
        self.reward_head = nn.Linear(h, cfg.num_support_bins)  # predict reward distribution over support

        # f_theta : latent state  ->  (policy logits, value)
        self.pred_body = nn.Sequential(nn.Linear(h, h), nn.ReLU())
        self.policy_head = nn.Linear(h, cfg.num_actions)
        self.value_head = nn.Linear(h, cfg.num_support_bins)  # predict value distribution over support

    def _repr(self, obs):
        return self.representation(obs)

    def _dyn(self, state, action):
        a = F.one_hot(action, self.num_actions).float()
        x = torch.cat([state, a], dim=-1)
        nxt = self.dynamics(x)
        reward = self.reward_head(nxt).squeeze(-1)
        return nxt, reward

    def _pred(self, state):
        x = self.pred_body(state)
        return self.policy_head(x), self.value_head(x).squeeze(-1)

    def initial_inference(self, obs):
        """Root of the search: encode the real observation, then evaluate it."""
        s = self._repr(obs)
        policy_logits, value_logits = self._pred(s)
        return s, policy_logits, value_logits

    def recurrent_inference(self, state, action):
        """One imagined step: advance the latent state, predict reward, evaluate."""
        nxt, reward = self._dyn(state, action)
        policy_logits, value_logits = self._pred(nxt)
        return nxt, reward, policy_logits, value_logits


# ----------------------------------------------------------------------------
# MCTS
# ----------------------------------------------------------------------------
class MinMaxStats:
    """Tracks min/max Q seen in the tree so Q can be normalized to [0,1] before
    it is compared against the exploration term in PUCT (gotcha #3)."""
    def __init__(self):
        self.maximum = -float("inf")
        self.minimum = float("inf")

    def update(self, value: float):
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)

    def normalize(self, value: float) -> float:
        if self.maximum > self.minimum:
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value


class Node:
    def __init__(self, prior: float):
        self.prior = prior            # P(s,a) from the parent's policy
        self.visit_count = 0          # N(s,a)
        self.value_sum = 0.0          # W(s,a)
        self.reward = 0.0             # predicted reward on the edge INTO this node
        self.hidden_state = None      # latent state s (set when expanded)
        self.children = {}            # action -> Node

    def expanded(self) -> bool:
        return len(self.children) > 0

    def value(self) -> float:         # Q(s,a) = W / N
        return 0.0 if self.visit_count == 0 else self.value_sum / self.visit_count


# ======================  TODO(you) #1  =======================================
def ucb_score(cfg: Config, parent: Node, child: Node, min_max: MinMaxStats) -> float:
    """Return the PUCT score for `child`, used during SELECTION to decide which
    child to descend into.

    Recall the shape we derived:

        score = Q(s,a)  +  U(s,a)

    where the exploit term is the child's (normalized) value, and the explore
    term is roughly:

        c * P(s,a) * sqrt( sum_b N(s,b) ) / ( 1 + N(s,a) )

    Ingredients available to you:
        child.prior         -> P(s,a)
        child.visit_count   -> N(s,a)
        parent.visit_count  -> sum_b N(s,b)   (the parent has been visited once
                                               per child-visit, so this equals
                                               the total child visits)
        child.value()       -> Q(s,a)   (remember to pass through
                                         min_max.normalize(...) so Q is on the
                                         same [0,1] scale as the explore term)
        child.reward        -> the predicted reward on the edge into child; the
                               "value" of taking this edge is really
                               reward + discount * child.value()

    A good first version can use the simple AlphaZero exploration term above with
    c ~ 1.25. If you want the exact paper version, the constant is replaced by:
        c1 + log( (sum_b N(s,b) + c2 + 1) / c2 )
    """
    Q = child.value() 
    Q = min_max.normalize(Q)
    U = cfg.c1 * child.prior * np.sqrt(parent.visit_count) / (1 + child.visit_count) # TODO: replace c1 with full expression 
    return Q + U

# =============================================================================


def select_child(cfg, node, min_max):
    """Pick the child with the highest PUCT score."""
    best_score, best_action, best_child = -float("inf"), None, None
    for action, child in node.children.items():
        score = ucb_score(cfg, node, child, min_max)
        if score > best_score:
            best_score, best_action, best_child = score, action, child
    return best_action, best_child


def expand_node(cfg, node, policy_logits, hidden_state, reward=0.0):
    """Attach a latent state to `node` and create one child per action, each
    seeded with its prior probability from the network's policy."""
    node.hidden_state = hidden_state
    node.reward = reward
    priors = torch.softmax(policy_logits, dim=-1).squeeze(0)
    for a in range(cfg.num_actions):
        node.children[a] = Node(prior=priors[a].item())


def add_exploration_noise(cfg, node):
    """Dirichlet noise on the ROOT priors only -> exploration during self-play."""
    actions = list(node.children.keys())
    noise = np.random.dirichlet([cfg.dirichlet_alpha] * len(actions))
    frac = cfg.root_exploration_frac
    for a, n in zip(actions, noise):
        node.children[a].prior = node.children[a].prior * (1 - frac) + n * frac


# ======================  TODO(you) #2  =======================================
def backpropagate(cfg: Config, search_path: List[Node], value: float,
                  min_max: MinMaxStats):
    """Propagate the leaf's value `value` back UP the visited path.

    Walk the path from the leaf to the root. At each node:
        - add the running value to node.value_sum   (W += G)
        - increment node.visit_count                 (N += 1)
        - update min_max with the node's new Q so PUCT normalization stays current
        - fold in this node's edge reward and discount before moving up one level:
              value  <-  node.reward + cfg.discount * value
          (this is exactly the "G <- r + gamma*G" step from the backup formula)

    Hint: iterate over `reversed(search_path)`.
    """
    for node in reversed(search_path):
        node.value_sum += value
        node.visit_count += 1
        Q = node.value()
        min_max.update(Q)
        value = node.reward + cfg.discount * value
# =============================================================================


def run_mcts(cfg, root, net, min_max):
    """Run `num_simulations` simulations to grow the tree rooted at `root`.
    Assumes `root` is already expanded (see expand_root)."""
    for _ in range(cfg.num_simulations):
        node = root
        search_path = [node]
        action = None

        # 1. SELECTION: descend via PUCT until we reach an unexpanded node
        while node.expanded():
            action, node = select_child(cfg, node, min_max)
            search_path.append(node)

        # 2. EXPANSION + EVALUATION: use the MODEL (g then f) to imagine the
        #    next latent state from the PARENT's state + the chosen action.
        parent = search_path[-2]
        with torch.no_grad():
            next_state, reward_logits, policy_logits, value_logits = net.recurrent_inference(
                parent.hidden_state, torch.tensor([action])
            )
        value = support_to_scalar(value_logits, cfg)  # convert value distribution to scalar
        reward = support_to_scalar(reward_logits, cfg)  # convert reward distribution to scalar
        expand_node(cfg, node, policy_logits, next_state, reward=reward.item())

        # 3. BACKUP
        backpropagate(cfg, search_path, value.item(), min_max)


def expand_root(cfg, net, obs):
    root = Node(prior=0.0)
    with torch.no_grad():
        state, policy_logits, _ = net.initial_inference(obs)
    expand_node(cfg, root, policy_logits, state, reward=0.0)
    add_exploration_noise(cfg, root)
    return root


# ----------------------------------------------------------------------------
# Self-play & replay buffer
# ----------------------------------------------------------------------------
@dataclass
class GameHistory:
    observations: list = field(default_factory=list)  # obs[i]
    actions: list = field(default_factory=list)        # action taken from obs[i]
    rewards: list = field(default_factory=list)        # reward for that transition
    policies: list = field(default_factory=list)       # MCTS policy at obs[i]  (pi)
    root_values: list = field(default_factory=list)    # MCTS root value at obs[i] (nu)

    def store(self, obs, action, reward, policy, root_value):
        self.observations.append(np.asarray(obs, dtype=np.float32))
        self.actions.append(int(action))
        self.rewards.append(float(reward))
        self.policies.append(np.asarray(policy, dtype=np.float32))
        self.root_values.append(float(root_value))


def select_action(visit_counts: np.ndarray, temperature: float) -> int:
    """Choose the real move from the root visit counts N(a)  (NOT from PUCT)."""
    if temperature == 0:
        return int(np.argmax(visit_counts))
    logits = visit_counts ** (1.0 / temperature)
    probs = logits / logits.sum()
    return int(np.random.choice(len(probs), p=probs))


def play_game(cfg, net) -> GameHistory:
    env = gym.make(cfg.env_id)
    obs, _ = env.reset()
    game = GameHistory()
    done = False
    step = 0
    while not done:
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        root = expand_root(cfg, net, obs_t)
        run_mcts(cfg, root, net, MinMaxStats())

        visit_counts = np.array(
            [root.children[a].visit_count for a in range(cfg.num_actions)],
            dtype=np.float32,
        )
        policy = visit_counts / visit_counts.sum()
        temperature = 1.0 if step < 30 else 0.0   # explore early, then greedy
        action = select_action(visit_counts, temperature)

        next_obs, reward, term, trunc, _ = env.step(action)
        done = term or trunc
        game.store(obs, action, reward, policy, root.value())
        obs = next_obs
        step += 1
    env.close()
    return game


# ======================  TODO(you) #3  =======================================
def compute_value_target(game: GameHistory, index: int, cfg: Config) -> float:
    """The n-step return value target  z_index :

        z = sum_{i=0}^{n-1}  gamma^i * reward[index + i]
              +  gamma^n * root_value[index + n]         (the bootstrap tail)

    Details:
        - n is cfg.td_steps, gamma is cfg.discount.
        - Only add reward[index + i] while (index + i) is a valid index into
          game.rewards (i.e. hasn't run past the end of the episode).
        - Only add the bootstrap term if (index + n) is a valid index into
          game.root_values; if the episode already ended, there is nothing to
          bootstrap from, so the tail is 0.
    """
    z = 0.0
    for i in range(cfg.td_steps):
        r_idx = i + index
        if r_idx < len(game.rewards):
            z += game.rewards[r_idx] * cfg.discount**(i)
        else:
            # doesn't matter because we dont use these per Line 412 in make_sample()
            # just need to make sure we dont over-index the rewards array 
            break
    boot_idx = cfg.td_steps + index
    if boot_idx < len(game.root_values):
        z += game.root_values[boot_idx] * cfg.discount**cfg.td_steps
    return z
# =============================================================================


# ----------------------------------------------------------------------------
# Data Generation
# ----------------------------------------------------------------------------

@dataclass
class Sample:
    obs: torch.Tensor
    actions: list                 # length K
    target_values: list           # length K+1
    target_rewards: list          # length K+1
    target_policies: list         # length K+1  (each an array over actions)


def make_sample(cfg, game, t) -> Sample:
    obs = torch.tensor(game.observations[t], dtype=torch.float32)

    actions = []
    for k in range(cfg.num_unroll_steps):
        idx = t + k
        actions.append(game.actions[idx] if idx < len(game.actions)
                       else random.randrange(cfg.num_actions))

    tv, tr, tp = [], [], []
    for k in range(cfg.num_unroll_steps + 1):
        cur = t + k
        tv.append(compute_value_target(game, cur, cfg))
        # reward target for unroll step k>=1 is the reward received on the
        # transition out of obs[t+k-1]; there is no reward prediction at the root.
        if k == 0:
            tr.append(0.0)
        else:
            r_idx = t + k - 1
            tr.append(game.rewards[r_idx] if r_idx < len(game.rewards) else 0.0)
        if cur < len(game.policies):
            tp.append(game.policies[cur])
        else:
            tp.append(np.ones(cfg.num_actions, dtype=np.float32) / cfg.num_actions)
    return Sample(obs, actions, tv, tr, tp)


class ReplayBuffer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.games: List[GameHistory] = []

    def save(self, game):
        self.games.append(game)
        if len(self.games) > self.cfg.replay_capacity:
            self.games.pop(0)

    def sample_batch(self) -> List[Sample]:
        batch = []
        for _ in range(self.cfg.batch_size):
            game = random.choice(self.games)
            t = random.randrange(len(game.actions))   # a position that has an action
            batch.append(make_sample(self.cfg, game, t))
        return batch


# ----------------------------------------------------------------------------
# Training  (unroll K steps along the REAL actions, match to targets)
# ----------------------------------------------------------------------------
def scale_gradient(tensor, scale):
    """Let `scale` fraction of the gradient flow through; identity on the forward
    pass. Used to down-weight the gradient entering the dynamics function."""
    return tensor * scale + tensor.detach() * (1 - scale)

def scaling_transform(x, cfg):
    return torch.sign(x) * (torch.sqrt(torch.abs(x)+1) - 1) + cfg.epsilon * x

def inverse_scaling_transform(y, cfg):
    # sign(y) * ( ((sqrt(1 + 4*eps*(|y| + 1 + eps)) - 1)/(2*eps))**2 - 1 )
    tmp = (torch.sqrt(1 + 4 * cfg.epsilon * (torch.abs(y) + 1 + cfg.epsilon)) - 1) / (2 * cfg.epsilon)
    return torch.sign(y) * (tmp ** 2 - 1)

def scalar_to_support(x, cfg):
    # transform the scalar values using the MuZero scaling transform
    x = scaling_transform(x, cfg)
    # define support bounds centered at zero
    support_min = cfg.support_min
    support_max = cfg.support_max
    support_bins = support_max - support_min + 1  # total number of bins in the support

    # clip transformed values to the support range
    x = x.clamp(support_min, support_max)
    # flatten the tensor so we can operate on all elements uniformly
    x_flat = x.reshape(-1)

    # lower support bin index for each value
    lower = x_flat.floor()
    # upper support bin is one above lower, capped by the max support
    upper = torch.clamp(lower + 1, max=support_max)

    # convert support values to zero-based tensor indices
    lower_idx = (lower - support_min).long()
    upper_idx = (upper - support_min).long()

    # fraction assigned to the upper bin
    upper_weight = x_flat - lower
    # fraction assigned to the lower bin
    lower_weight = 1.0 - upper_weight

    # initialize a zero distribution for each scalar value
    dist = torch.zeros((x_flat.shape[0], support_bins), device=x.device, dtype=x.dtype)
    # add lower-bin weights into the distribution
    dist.scatter_add_(1, lower_idx.unsqueeze(1), lower_weight.unsqueeze(1))
    # add upper-bin weights into the distribution
    dist.scatter_add_(1, upper_idx.unsqueeze(1), upper_weight.unsqueeze(1))

    # reshape back to the original input shape plus the support dimension
    return dist.view(*x.shape, support_bins)

def support_to_scalar(logits, cfg):
    # convert logits to probabilities using softmax
    probs = torch.softmax(logits, dim=-1)
    # define support bounds centered at zero
    support_min = cfg.support_min
    support_max = cfg.support_max
    # create a tensor of support values corresponding to each bin
    support_values = torch.arange(support_min, support_max + 1, device=logits.device, dtype=logits.dtype)
    # compute the expected value by summing over the
    expected_value = (probs * support_values).sum(dim=-1)
    # apply the inverse scaling transform to map back to the original scalar range
    return inverse_scaling_transform(expected_value, cfg)


def soft_cross_entropy(logits, target_dist):
    logp = F.log_softmax(logits, dim=-1) # log(softmax(logits)) - so log(probability distribution),i.e. "surprisal" of outcome i under predicted distribution; in (-∞, 0]
    return -(target_dist * logp).sum(dim=-1).mean() # weight surprisal by how often outcomes "actually" occur (target distr.)


def update_weights(cfg, net, optimizer, batch: List[Sample]):
    obs = torch.stack([b.obs for b in batch])                       # (B, obs_dim)
    actions = torch.tensor([b.actions for b in batch])              # (B, K)
    tgt_v = torch.tensor([b.target_values for b in batch])          # (B, K+1)
    tgt_r = torch.tensor([b.target_rewards for b in batch])         # (B, K+1)
    tgt_p = torch.tensor(np.array([b.target_policies for b in batch]))  # (B, K+1, A)

    # k = 0 : root (representation + prediction), no reward prediction
    state, policy_logits, value_logits = net.initial_inference(obs)

    loss = (
        soft_cross_entropy(policy_logits, tgt_p[:, 0]) 
        + 0.25 * soft_cross_entropy(value_logits, scalar_to_support(tgt_v[:, 0], cfg))  # bring the target into the network's space
        
    )

    # k = 1..K : imagined steps along the real action sequence
    for k in range(cfg.num_unroll_steps):
        state, reward_logits, policy_logits, value_logits = net.recurrent_inference(state, actions[:, k])
        state = scale_gradient(state, 0.5)
        step_loss = (
            soft_cross_entropy(policy_logits, tgt_p[:, k + 1])
            + 0.25 * soft_cross_entropy(value_logits, scalar_to_support(tgt_v[:, k + 1], cfg))  # bring the target into the network's space
            + soft_cross_entropy(reward_logits, scalar_to_support(tgt_r[:, k + 1], cfg))  # bring the target into the network's space
        )
        loss = loss + step_loss / cfg.num_unroll_steps

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


# ----------------------------------------------------------------------------
# Main loop
# ----------------------------------------------------------------------------
def main():
    cfg = Config()
    net = MuZeroNet(cfg)
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.lr, weight_decay=1e-4)
    buffer = ReplayBuffer(cfg)

    for it in range(cfg.num_iterations):
        # --- self-play ---
        lengths = []
        for _ in range(cfg.games_per_iteration):
            game = play_game(cfg, net)
            buffer.save(game)
            lengths.append(len(game.actions))   # CartPole reward == episode length

        # --- train ---
        losses = []
        for _ in range(cfg.train_steps_per_iteration):
            losses.append(update_weights(cfg, net, optimizer, buffer.sample_batch()))

        print(f"iter {it:3d} | avg episode length {np.mean(lengths):6.1f} "
              f"| loss {np.mean(losses):.3f}")


if __name__ == "__main__":
    if gym is None:
        raise SystemExit("Please `pip install gymnasium torch numpy` first.")
    main()