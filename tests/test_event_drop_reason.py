"""D17: a data-quality loss must not be filed as a design decision.

Measured on MAG-50 r5: 3 of 3 `extends` relations carried no
`difference_dimension`, so ``relation_to_event_type`` returned ``None`` and the
audit entry said ``not_an_evolution_event`` — the same reason code used for
``parallel`` / ``not_comparable``, which genuinely produce no events by design.

That collapsection is dishonest in the one place that exists to be honest: the
ignored-events audit trail. A reader (or a later fix) cannot tell apart
"deliberately no event" from "should have been an event, but the model omitted a
field". The fix is to give each drop its own reason code.
"""

from __future__ import annotations

from academic_landscape.timeline_events import (
    relation_drop_reason,
    relation_to_event_type,
)


def test_design_non_events_keep_their_reason() -> None:
    for relation_type in ("parallel", "not_comparable", "unknown"):
        assert relation_drop_reason(relation_type, None) == "not_an_evolution_event"


def test_extends_without_dimension_is_a_data_loss_not_a_design_decision() -> None:
    reason = relation_drop_reason("extends", None)
    assert reason == "missing_difference_dimension"


def test_extends_with_unknown_dimension_is_named_separately() -> None:
    """An unknown dimension is also a data problem, not a policy decision."""
    assert relation_drop_reason("extends", "vibes") == "unrecognised_difference_dimension"


def test_extends_with_known_dimension_produces_an_event() -> None:
    for dimension in ("object", "conclusion", "method"):
        assert relation_to_event_type("extends", dimension) is not None
        assert relation_drop_reason("extends", dimension) is None


def test_reason_matches_the_event_mapping_consistently() -> None:
    """A relation with no event must have a reason, and vice versa."""
    cases = [
        ("parallel", None),
        ("extends", None),
        ("extends", "object"),
        ("extends", "vibes"),
        ("supports", None),
        ("challenges", None),
    ]
    for relation_type, dimension in cases:
        has_event = relation_to_event_type(relation_type, dimension) is not None
        reason = relation_drop_reason(relation_type, dimension)
        assert has_event == (reason is None), (relation_type, dimension, reason)
