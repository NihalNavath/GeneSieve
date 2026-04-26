from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

try:
    from models import GenesieveAction, GenesieveObservation
except:
    from .models import GenesieveAction, GenesieveObservation


class GenesieveEnv(
    EnvClient[GenesieveAction, GenesieveObservation, State]
):
    """
    Client for the GeneSieve Environment.
    """

    def _step_payload(self, action: GenesieveAction) -> Dict:
        return {
            "tool": action.tool,
            "args": action.args,
        }

    def _parse_result(self, payload: Dict) -> StepResult[GenesieveObservation]:
        # The server serializes Observation via serialize_observation() which
        # produces: {"observation": {obs_fields}, "reward": ..., "done": ...}
        obs_data = payload.get("observation", {})

        observation = GenesieveObservation(
            organism=obs_data["organism"],
            budget_remaining=obs_data["budget_remaining"],
            genes_available=obs_data.get("genes_available", []),
            last_result=obs_data.get("last_result"),
            done=payload.get("done", False),
            reward=payload.get("reward", 0.0),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )