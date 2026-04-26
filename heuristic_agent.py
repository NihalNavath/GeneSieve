"""
GeneSieve RL Agent — Actor-Critic with Gene-Aware Policy

Key design decisions:
  1. Two-headed action selection: policy picks BOTH tool AND gene (not random)
  2. Per-gene knowledge tracking: agent remembers inspect/homolog/binding results
     within each episode and encodes them as features
  3. Prior scores from environment used for initial gene ranking
  4. Actor-Critic (A2C): value baseline reduces variance vs raw REINFORCE
  5. Gradient clipping + batched updates for stability

Fixes applied:
  - FIX 1: Tool-gene coupling — gene scoring is now conditioned on the sampled
            tool via a learned tool embedding, so submit_target learns to pick
            verified genes instead of random ones.
  - FIX 2: Entropy floor — entropy coefficient no longer decays to zero,
            preventing policy collapse into blind submits.
  - FIX 3: Separate advantages for tool and gene — gene choice gets a detached
            advantage so tool and gene aren't blamed equally for the same outcome.
  - FIX 4: Real entropy — uses Categorical.entropy() over full distribution
            instead of sampled log probs.
  - FIX 5: verified_score as 10th gene feature — explicit signal for how much
            positive evidence exists for each gene, making it easy to learn
            "submit the most verified gene."
  - FIX 6: has_verified_candidate in global features — binary signal so agent
            knows when a confirmed good target is ready.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from client import GenesieveEnv
from models import GenesieveAction


ACTIONS = [
    "inspect_gene",
    "check_human_homolog",
    "test_binding",
    "submit_target",
]

GLOBAL_DIM = 6      # global state features
GENE_FEAT_DIM = 10  # 3 priors + 3 test results + 3 tested flags + 1 verified_score
TOOL_EMBED_DIM = 16 # dimension of tool embedding
HIDDEN = 64


# ── Policy Network ──────────────────────────────────────────────────────────

class GeneAwarePolicy(nn.Module):
    """
    Two-headed policy that selects both TOOL and GENE.

    Architecture:
      Global features  → shared encoder → tool head (4 tools)
                                         → value head (baseline)
      Per-gene features → gene encoder  ─┐
      Global context ────────────────────┤
      Tool embedding  ────────────────────┤→ gene scoring → gene head
    """

    def __init__(self):
        super().__init__()

        # Shared encoder for global state
        self.global_enc = nn.Sequential(
            nn.Linear(GLOBAL_DIM, HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN),
            nn.ReLU(),
        )

        # Tool selection head (4 tools)
        self.tool_head = nn.Linear(HIDDEN, 4)

        # Per-gene feature encoder
        self.gene_enc = nn.Sequential(
            nn.Linear(GENE_FEAT_DIM, HIDDEN),
            nn.ReLU(),
        )

        # Tool embedding so gene scorer knows which tool was chosen
        self.tool_embed = nn.Embedding(4, TOOL_EMBED_DIM)

        # Gene scoring: combines global context + gene embedding + tool embedding
        self.gene_score = nn.Sequential(
            nn.Linear(HIDDEN * 2 + TOOL_EMBED_DIM, HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, 1),
        )

        # Value head (for A2C baseline)
        self.value_head = nn.Linear(HIDDEN, 1)

    def act(self, obs, knowledge):
        global_feats = encode_global(obs, knowledge)
        gene_feats = encode_genes(obs, knowledge)

        # Encode global state once
        g = self.global_enc(global_feats)   # [HIDDEN]

        # Tool selection
        tool_logits = self.tool_head(g)     # [4]
        tool_dist = torch.distributions.Categorical(logits=tool_logits)
        tool_idx = tool_dist.sample()

        # Value estimate
        value = self.value_head(g).squeeze(-1)

        # Gene selection conditioned on the sampled tool
        gene_logits = self._score_genes(g, gene_feats, tool_idx)
        gene_dist = torch.distributions.Categorical(logits=gene_logits)
        gene_idx = gene_dist.sample()

        tool = ACTIONS[tool_idx.item()]
        gene_name = obs.genes_available[gene_idx.item()]["gene_name"]

        action = GenesieveAction(
            tool=tool,
            args={"gene_name": gene_name}
        )

        return (
            action,
            tool_dist.log_prob(tool_idx),
            gene_dist.log_prob(gene_idx),
            value,
            tool_dist,   # FIX 4: return full distributions for real entropy
            gene_dist,
        )

    def _score_genes(self, g, gene_features, tool_idx):
        """Score each gene given global context and the chosen tool."""
        ge = self.gene_enc(gene_features)                             # [num_genes, HIDDEN]
        num_genes = gene_features.shape[0]

        g_expanded = g.unsqueeze(0).expand(num_genes, -1)             # [num_genes, HIDDEN]

        tool_emb = self.tool_embed(tool_idx)                          # [TOOL_EMBED_DIM]
        tool_expanded = tool_emb.unsqueeze(0).expand(num_genes, -1)   # [num_genes, TOOL_EMBED_DIM]

        combined = torch.cat([g_expanded, ge, tool_expanded], dim=-1) # [num_genes, HIDDEN*2+16]
        gene_logits = self.gene_score(combined).squeeze(-1)           # [num_genes]
        return gene_logits

    def forward(self, global_features, gene_features, tool_idx):
        """
        Args:
            global_features: [GLOBAL_DIM] tensor
            gene_features:   [num_genes, GENE_FEAT_DIM] tensor
            tool_idx:        scalar LongTensor — the chosen tool index

        Returns:
            tool_logits: [4]
            gene_logits: [num_genes]
            value:       scalar
        """
        g = self.global_enc(global_features)
        tool_logits = self.tool_head(g)
        value = self.value_head(g).squeeze(-1)
        gene_logits = self._score_genes(g, gene_features, tool_idx)
        return tool_logits, gene_logits, value


# ── Knowledge Tracking ──────────────────────────────────────────────────────

def update_knowledge(knowledge: dict, last_result: dict | None):
    """Update per-gene knowledge from the last tool result."""
    if last_result is None:
        return

    tool = last_result.get("tool")
    gene = last_result.get("gene")
    result = last_result.get("result")

    if result in ("already_tested", "invalid_gene"):
        return

    if gene not in knowledge:
        knowledge[gene] = {}

    if tool == "inspect_gene":
        knowledge[gene]["essential"] = result
    elif tool == "check_human_homolog":
        knowledge[gene]["safe"] = result        # True = no human homolog
    elif tool == "test_binding":
        knowledge[gene]["druggable"] = result
    elif tool == "submit_target":
        #print("   >>> SUBMIT <<<")
        knowledge[gene]["submitted"] = result


# ── Feature Encoding ────────────────────────────────────────────────────────

def encode_global(obs, knowledge: dict) -> torch.Tensor:
    """Encode global episode state into a feature vector."""

    # FIX 6: binary signal — does the agent have a fully confirmed candidate?
    has_verified_candidate = float(any(
        k.get("essential") is True and k.get("safe") is True
        for k in knowledge.values()
    ))

    tested_count = sum(1 for k in knowledge.values() if len(k) > 0)
    num_genes = max(len(obs.genes_available), 1)

    last = obs.last_result or {}
    result_val = last.get("result")
    if result_val is True:
        result_signal = 1.0
    elif result_val is False:
        result_signal = -1.0
    else:
        result_signal = 0.0

    return torch.tensor([
        obs.budget_remaining / 15.0,
        len(obs.genes_available) / 20.0,
        result_signal,
        obs.reward if obs.reward else 0.0,
        has_verified_candidate,           # FIX 6: was good_candidates / num_genes
        tested_count / num_genes,
    ], dtype=torch.float32)


def encode_genes(obs, knowledge: dict) -> torch.Tensor:
    """
    Encode per-gene features from priors + accumulated knowledge.

    For each gene, 10 features:
      [essential_score, safety_score, drug_likelihood,   <- env priors
       essential_result, safe_result, druggable_result,  <- test results
       inspected?, homolog_checked?, binding_tested?,    <- tested flags
       verified_score]                                   <- FIX 5: positive evidence
    """
    features = []
    for g in obs.genes_available:
        name = g["gene_name"]
        k = knowledge.get(name, {})

        ess = k.get("essential")
        safe = k.get("safe")
        drug = k.get("druggable")

        # FIX 5: explicit scalar ranking of positive evidence for this gene
        verified_score = sum([
            1.0 if ess is True else 0.0,
            1.0 if safe is True else 0.0,
            1.0 if drug is True else 0.0,
        ]) / 3.0

        features.append([
            g.get("essential_score", 0.5),
            g.get("safety_score", 0.5),
            g.get("drug_likelihood", 0.5),

            1.0 if ess is True else (-1.0 if ess is False else 0.0),
            1.0 if safe is True else (-1.0 if safe is False else 0.0),
            1.0 if drug is True else (-1.0 if drug is False else 0.0),

            1.0 if ess is not None else 0.0,
            1.0 if safe is not None else 0.0,
            1.0 if drug is not None else 0.0,

            verified_score,
        ])

    return torch.tensor(features, dtype=torch.float32)


# ── Episode Runner ──────────────────────────────────────────────────────────

def run_episode(env, policy):
    step = env.reset()
    obs = step.observation

    tool_lps, gene_lps, values, rewards = [], [], [], []
    tool_dists, gene_dists = [], []   # FIX 4: collect full distributions
    episode_trace = []

    knowledge = {}
    done = False
    t = 0

    while not done:
        action, tool_lp, gene_lp, value, tool_dist, gene_dist = policy.act(obs, knowledge)

        step = env.step(action)
        next_obs = step.observation

        reward = step.reward
        done = step.done

        update_knowledge(knowledge, next_obs.last_result)

        episode_trace.append({
            "t": t,
            "tool": action.tool,
            "gene": action.args.get("gene_name"),
            "result": next_obs.last_result,
            "reward": reward,
            "obs_reward": next_obs.reward,
            "budget": next_obs.budget_remaining,
            "done": done,
        })

        tool_lps.append(tool_lp)
        gene_lps.append(gene_lp)
        tool_dists.append(tool_dist)
        gene_dists.append(gene_dist)
        values.append(value)
        rewards.append(reward)

        obs = next_obs
        t += 1

    return tool_lps, gene_lps, values, rewards, episode_trace, tool_dists, gene_dists


# ── Training ────────────────────────────────────────────────────────────────

def compute_returns(rewards, gamma=0.99):
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return torch.tensor(returns)


REWARD_SCALE = 2.5


def train():
    policy = GeneAwarePolicy()
    optimizer = optim.Adam(policy.parameters(), lr=1e-4)

    NUM_EPISODES = 500
    BATCH_SIZE = 8

    running_baseline = 0.0
    running_reward = None

    with GenesieveEnv(base_url="http://localhost:8000").sync() as env:

        batch_loss = 0
        batch_count = 0

        for episode in range(NUM_EPISODES):
            tool_lps, gene_lps, values, rewards, episode_trace, tool_dists, gene_dists = run_episode(env, policy)

            scaled_rewards = [r / REWARD_SCALE for r in rewards]

            returns = compute_returns(scaled_rewards)
            values_t = torch.stack(values)

            advantages = returns - values_t.detach() - running_baseline

            # FIX 3: separate advantages — gene gets detached advantage
            policy_loss = 0
            for tlp, glp, adv in zip(tool_lps, gene_lps, advantages):
                policy_loss += -tlp * adv + -glp * adv.detach()

            # Value loss
            value_loss = F.mse_loss(values_t, returns)

            # FIX 2: entropy floor — never decays to zero
            entropy_coeff = max(0.005, 0.02 * (1.0 - episode / (NUM_EPISODES * 0.6)))

            # FIX 4: real entropy over full distributions
            entropy_bonus = sum(d.entropy() for d in tool_dists + gene_dists)

            loss = policy_loss + 0.5 * value_loss - entropy_coeff * entropy_bonus

            batch_loss = batch_loss + loss / BATCH_SIZE
            batch_count += 1

            if batch_count >= BATCH_SIZE:
                optimizer.zero_grad()
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                optimizer.step()
                batch_loss = 0
                batch_count = 0

            total_reward = sum(rewards)
            scaled_total = sum(scaled_rewards)

            running_baseline = 0.9 * running_baseline + 0.1 * scaled_total

            if running_reward is None:
                running_reward = total_reward
            else:
                running_reward = 0.95 * running_reward + 0.05 * total_reward

            if episode % 20 == 0:
                print("\n[TRACE]")
                for i, s in enumerate(episode_trace):
                    print(
                        f"{i:02d} | {s['tool']:>22} | "
                        f"{s['gene']:<10} | "
                        f"{s['result']} | "
                        f"{s['reward']:+.2f}"
                    )

            if episode % 10 == 0:
                print(
                    f"Episode {episode:3d} | "
                    f"Reward: {total_reward:+.3f} | "
                    f"Avg: {running_reward:+.3f}"
                )


if __name__ == "__main__":
    train()