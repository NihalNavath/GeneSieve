from typing import Dict, List, Optional, Literal
from openenv.core.env_server.types import Action, Observation
from pydantic import Field


# ── Action ────────────────────────────────────────────────────────────────

class GenesieveAction(Action):
    """
    Represents an agent action in the environment.
    """

    tool: Literal[
        "inspect_gene",
        "check_human_homolog",
        "test_binding",
        "submit_target",
    ] = Field(
        ...,
        description="Tool the agent wants to call"
    )

    args: Dict[str, str] = Field(
        default_factory=dict,
        description="Arguments for the tool (must include gene_name for most tools)"
    )


# ── Observation ────────────────────────────────────────────────────────────

class GeneInfo(Observation):
    """
    Lightweight gene representation exposed to the agent.
    (No hidden labels like essential/human_homolog)
    """

    gene_name: str = Field(..., description="Gene identifier")
    function: Optional[str] = Field(None, description="Gene function")
    pathway: Optional[str] = Field(None, description="Biological pathway")


class GenesieveObservation(Observation):
    """
    Observation returned to the agent at each step.

    Inherits from Observation which provides:
      - done: bool
      - reward: bool | int | float | None
      - metadata: Dict[str, Any]
    """

    organism: str = Field(..., description="Name of organism")

    budget_remaining: int = Field(
        ...,
        description="Remaining number of experiments"
    )

    genes_available: List[Dict] = Field(
        ...,
        description="List of visible genes"
    )

    last_result: Optional[dict] = Field(
        default=None,
        description="Result of the last tool call"
    )