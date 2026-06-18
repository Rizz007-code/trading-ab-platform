# api/routers/strategies.py
"""
GET /api/v1/strategies

Endpoints:
  GET /          — list all registered strategies with metadata
  GET /{name}    — single strategy info by class name
"""

from fastapi import APIRouter, HTTPException

from api.schemas import StrategyOut
from strategies.strategy_a import MACrossoverStrategy
from strategies.strategy_b import MARSIStrategy
from strategies.strategy_c import MACDStrategy

router = APIRouter()

# ── Strategy registry ─────────────────────────────────────────────────────────
# Maps the public-facing class name to the strategy class.
# Add new strategies here — they'll appear in all endpoints automatically.

STRATEGY_REGISTRY: dict = {
    "MACrossoverStrategy": MACrossoverStrategy,
    "MARSIStrategy":       MARSIStrategy,
    "MACDStrategy":        MACDStrategy,
}


def _strategy_to_schema(class_name: str) -> StrategyOut:
    cls      = STRATEGY_REGISTRY[class_name]
    instance = cls()
    info     = instance.info
    return StrategyOut(
        name        = info.name,
        version     = info.version,
        description = info.description,
        parameters  = info.parameters,
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get(
    "/",
    response_model = list[StrategyOut],
    summary        = "List all available strategies",
)
def list_strategies() -> list[StrategyOut]:
    """Return metadata for every registered trading strategy."""
    return [_strategy_to_schema(name) for name in STRATEGY_REGISTRY]


@router.get(
    "/{strategy_name}",
    response_model = StrategyOut,
    summary        = "Get strategy details by class name",
    responses      = {404: {"description": "Strategy not found"}},
)
def get_strategy(strategy_name: str) -> StrategyOut:
    """
    Return metadata for a single strategy.

    `strategy_name` must be one of the class names returned by GET /.
    """
    if strategy_name not in STRATEGY_REGISTRY:
        raise HTTPException(
            status_code = 404,
            detail      = {
                "error":     f"Strategy '{strategy_name}' not found.",
                "available": list(STRATEGY_REGISTRY.keys()),
            },
        )
    return _strategy_to_schema(strategy_name)
