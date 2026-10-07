"""Convert analysis objects into pure JSON-safe data.

Why this module exists (audit finding D9)
-----------------------------------------
``dataclasses.asdict`` recurses only into dataclasses. The project mixes both
dataclasses and Pydantic models, so relations (``PaperRelation``) and evolution
events (``EvolutionEvent``) passed straight through ``asdict`` and were then
stringified by ``json.dumps(..., default=str)`` into Python reprs. The artifact
looked populated while being unreadable to any program — the worst possible
failure mode for an API-facing product.

Rule: structured output must survive ``json.dumps`` with **no** ``default=``
fallback. Unknown types raise ``TypeError`` instead of degrading silently; a
loud failure is recoverable, a repr masquerading as data is not.
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from enum import Enum
from typing import Any

_PRIMITIVES = (str, int, float, bool)


def to_jsonable(value: Any) -> Any:
    """Recursively convert ``value`` into JSON-safe primitives/containers."""
    if value is None or isinstance(value, _PRIMITIVES):
        return value

    # Pydantic v2 (BaseModel and pydantic dataclasses both expose model_dump).
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            dumped = model_dump(mode="json")
        except TypeError:  # pragma: no cover - defensive for non-v2 shims
            dumped = model_dump()
        return to_jsonable(dumped)

    if isinstance(value, Enum):
        return to_jsonable(value.value)

    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: to_jsonable(getattr(value, field.name)) for field in fields(value)}

    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}

    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]

    if isinstance(value, (set, frozenset)):
        # Sorted so repeated runs produce byte-identical artifacts (14.3 可复现).
        return [to_jsonable(item) for item in sorted(value, key=repr)]

    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", errors="replace")

    raise TypeError(
        f"{type(value).__name__} is not JSON-serializable; "
        "add an explicit conversion in serialization.to_jsonable rather than "
        "letting it degrade to a repr string"
    )
