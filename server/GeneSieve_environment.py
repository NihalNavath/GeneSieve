from uuid import uuid4
import json
import os
import random
from typing import Any, Optional

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import GenesieveAction, GenesieveObservation
except ImportError:
    from models import GenesieveAction, GenesieveObservation


BUDGET = 15
MAX_GENES_SHOWN = 20
MIN_VALID_VISIBLE = 3  # guarantee at least this many valid targets in view
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
ORGANISMS = ["ecoli", "saureus", "mtb"]

# Ground-truth fields hidden from the agent
HIDDEN_GENE_FIELDS = {"binding_compounds", "has_human_homolog", "is_valid_target", "essential"}

# Noise magnitude for prior scores (higher = harder to exploit without testing)
PRIOR_NOISE = 0.20


class GenesieveEnvironment(Environment):
    def __init__(self):
        super().__init__()
        self.env_id = str(uuid4())
        self._rng = random.Random()
        self._state = None
        self._gene_db = {}
        self._load_databases()

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> GenesieveObservation:
        """Reset environment and return initial observation."""

        if seed is not None:
            self._rng = random.Random(seed)

        self.env_id = episode_id or str(uuid4())

        key = self._rng.choice(ORGANISMS)
        db = self._gene_db[key]

        all_genes = db["genes"]
        lookup = {g["gene_name"]: g for g in all_genes}

        # ── Structured sampling: guarantee learnable episodes ────────────
        valid = [g for g in all_genes if g["is_valid_target"]]
        invalid = [g for g in all_genes if not g["is_valid_target"]]

        # Ensure at least MIN_VALID_VISIBLE valid targets are visible
        n_valid = min(MIN_VALID_VISIBLE, len(valid))
        n_fill = min(MAX_GENES_SHOWN - n_valid, len(invalid))

        chosen_valid = self._rng.sample(valid, n_valid)
        chosen_invalid = self._rng.sample(invalid, n_fill)

        # Fill remaining slots from whatever's left
        remaining = [g for g in all_genes
                     if g not in chosen_valid and g not in chosen_invalid]
        n_remaining = MAX_GENES_SHOWN - n_valid - n_fill
        if n_remaining > 0 and remaining:
            chosen_extra = self._rng.sample(remaining, min(n_remaining, len(remaining)))
        else:
            chosen_extra = []

        visible = chosen_valid + chosen_invalid + chosen_extra
        self._rng.shuffle(visible)

        self._state = {
            "organism": db["display_name"],
            "visible_genes": visible,
            "all_genes": lookup,
            "budget": BUDGET,
            "tested": set(),
            "done": False,
            "cumulative_reward": 0.0,
            "last_result": None,
            "history": [],
            "step_count": 0,
        }

        obs = GenesieveObservation(
            organism=self._state["organism"],
            budget_remaining=self._state["budget"],
            genes_available=self._prepare_visible(visible),
            last_result=None,
            done=False,
            reward=0.0,
        )

        return obs

    def step(
        self,
        action: GenesieveAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> GenesieveObservation:
        """Execute an action and return the resulting observation."""

        if self._state is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        if self._state["done"]:
            return self._build_observation(step_reward=-0.5)

        tool = action.tool
        gene_name = action.args.get("gene_name", "")
        reward = 0.0
        result = None

        key = (tool, gene_name)

        valid_tools = {
            "inspect_gene",
            "check_human_homolog",
            "test_binding",
            "submit_target",
        }

        # Validate tool + gene independently

        invalid_tool = tool not in valid_tools
        invalid_gene = gene_name not in self._state["all_genes"]

        if invalid_tool or invalid_gene:
            if invalid_tool and invalid_gene:
                result = "invalid_tool_and_gene"
                reward = -0.4
            elif invalid_tool:
                result = "invalid_tool"
                reward = -0.3
            else:
                result = "invalid_gene"
                reward = -0.3

            self._state["last_result"] = {
                "tool": tool,
                "gene": gene_name,
                "result": result,
            }

            self._state["cumulative_reward"] += reward
            self._state["step_count"] += 1
            return self._build_observation(step_reward=reward)

        g = self._state["all_genes"][gene_name]

        # repeated action penalty
        if key in self._state["tested"]:
            reward = -0.15
            result = "already_tested"
        else:
            self._state["tested"].add(key)

            ## The three tools that the agent can use TODO: add another one that does both at once?
            # inspect: reveals whether gene is essential
            if tool == "inspect_gene":
                result = g["essential"]
                reward = 0.5 if result else -0.2

            # human homolog check: reveals safety info
            elif tool == "check_human_homolog":
                has_homolog = g["has_human_homolog"]
                result = not has_homolog  # True = safe (no human homolog)
                reward = 0.3 if result else -0.1

            # binding test: reveals druggability
            elif tool == "test_binding":
                has_binding = len(g["binding_compounds"]) > 0
                result = has_binding
                reward = 0.35 if result else -0.1

            # submit target: ends the episode
            elif tool == "submit_target":
                self._state["done"] = True
                is_valid = g["is_valid_target"]
                result = is_valid

                tools_used = {t for (t, gn) in self._state["tested"] if gn == gene_name}
                num_tests = len(tools_used)

                signals = []
                for entry in self._state.get("history", []):
                    if entry["gene"] == gene_name:
                        signals.append(entry["result"])

                efficiency = max(0, (BUDGET - self._state["step_count"]) / BUDGET)

                # Reward logic
                if is_valid:
                    if num_tests == 0:
                        reward = -0.5  # no reward for blind luck

                    elif num_tests == 1:
                        reward = 0.4  # weak evidence → capped reward

                    else:
                        reward = min(2.0 + 0.5 * num_tests + 0.5 * efficiency, 3.0)

                else:
                    reward = -1.0 - 0.3 * num_tests  # more confident → more penalty

                    if num_tests == 0:
                        reward -= 0.5

                # Penalize inconsistent evidence
                if signals:
                    if signals.count(False) > signals.count(True):
                        reward -= 0.5  # ignored negative evidence

            # Store result for observation
            self._state["last_result"] = {
                "tool": tool,
                "gene": gene_name,
                "result": result,
            }

            if "history" not in self._state:
                    self._state["history"] = []

            self._state["history"].append({
                "tool": tool,
                "gene": gene_name,
                "result": result,
            })

            self._state["budget"] -= 1
            self._state["step_count"] += 1

        # Budget exhausted without submission → softer penalty
        if self._state["budget"] <= 0 and not self._state["done"]:
            self._state["done"] = True
            reward -= 0.5  # penalize but don't crush exploration

        self._state["cumulative_reward"] += reward
        return self._build_observation(step_reward=reward)

    @property
    def state(self) -> State:
        """Return current environment state (required by OpenEnv)."""
        if self._state is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        return State(
            episode_id=self.env_id,
            step_count=self._state["step_count"],
        )

    # Helpers

    def _build_observation(self, step_reward: float) -> GenesieveObservation:
        """Build an observation from current state."""
        return GenesieveObservation(
            organism=self._state["organism"],
            budget_remaining=self._state["budget"],
            genes_available=self._prepare_visible(self._state["visible_genes"]),
            last_result=self._state.get("last_result"),
            done=self._state["done"],
            reward=step_reward,
        )

    def _prepare_visible(self, genes: list) -> list:
        """
        Prepare gene data for the agent: hide ground truth but add noisy priors.

        Prior scores are weak, noisy signals that correlate with the true labels.
        This lets the agent rank genes before spending actions, shifting behavior
        from "probe randomly" to "predict → rank → confirm".
        """
        result = []
        for g in genes:
            entry = {k: v for k, v in g.items() if k not in HIDDEN_GENE_FIELDS}
            entry["essential_score"] = self._noisy_prior(g["essential"], 0.7, 0.25)
            entry["safety_score"] = self._noisy_prior(not g["has_human_homolog"], 0.7, 0.25)
            entry["drug_likelihood"] = self._noisy_prior(
                len(g["binding_compounds"]) > 0, 0.65, 0.3
            )

            result.append(entry)
        return result

    def _noisy_prior(self, truth: bool, high_center: float, low_center: float) -> float:
        """
        Generate a noisy score that correlates with ground truth.

        If truth=True:  score centered around high_center (e.g. 0.7)
        If truth=False: score centered around low_center  (e.g. 0.25)
        Plus Gaussian noise, clamped to [0, 1].
        """
        center = high_center if truth else low_center
        noise = self._rng.gauss(0, PRIOR_NOISE)
        return max(0.0, min(1.0, center + noise))

    def _load_databases(self):
        for key in ORGANISMS:
            path = os.path.join(DATA_DIR, f"genes_{key}.json")
            with open(path) as f:
                self._gene_db[key] = json.load(f)