"""JSON-compatible projection of the canonical model."""

import json
import math
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from datetime import date, datetime
from numbers import Integral, Real
from typing import Any

from ..models.evidence import CategoricalContext


class _CircularReferenceError(ValueError):
    """Internal signal raised when normalization meets a cycle; surfaced as ValueError."""


def _circular_reference_error(path: str) -> _CircularReferenceError:
    location = f" at '{path}'" if path else ""
    return _CircularReferenceError(
        f"Cannot export circular reference{location}: the sampled values contain a cycle. "
        "Retry with include_values=False (the default) to omit sampled values."
    )


def _normalize(
    value: Any,
    include_values: bool = False,
    *,
    evidence_details: bool = False,
    _active: set[int] | None = None,
    _path: str = "",
) -> Any:
    # Track container identities along the current traversal path only, so a value
    # shared by separate branches (acyclic) is still exported, while a value that
    # refers back to itself raises a clear error instead of recursing forever.
    if _active is None:
        _active = set()
    if is_dataclass(value):
        if id(value) in _active:
            raise _circular_reference_error(_path)
        _active.add(id(value))
        try:
            result: dict[str, Any] = {}
            for field in fields(value):
                if not include_values and (
                    field.name == "top_values"
                    or (isinstance(value, CategoricalContext) and field.name == "values")
                ):
                    result[field.name] = []
                else:
                    result[field.name] = _normalize(
                        getattr(value, field.name), include_values,
                        evidence_details=(type(value).__name__ == "Evidence" and field.name == "details"),
                        _active=_active, _path=f"{_path}.{field.name}" if _path else field.name,
                    )
            return result
        finally:
            _active.discard(id(value))
    if isinstance(value, Mapping):
        if id(value) in _active:
            raise _circular_reference_error(_path)
        _active.add(id(value))
        try:
            result = {}
            for key, item in value.items():
                if evidence_details and not include_values and isinstance(item, str):
                    continue
                if evidence_details and not include_values and isinstance(item, (tuple, list)) and any(isinstance(v, str) for v in item):
                    continue
                result[str(key)] = _normalize(
                    item, include_values, evidence_details=evidence_details,
                    _active=_active, _path=f"{_path}[{key!r}]",
                )
            return result
        finally:
            _active.discard(id(value))
    if isinstance(value, (tuple, list)):
        if id(value) in _active:
            raise _circular_reference_error(_path)
        _active.add(id(value))
        try:
            return [
                _normalize(
                    item, include_values, evidence_details=evidence_details,
                    _active=_active, _path=f"{_path}[{index}]",
                )
                for index, item in enumerate(value)
            ]
        finally:
            _active.discard(id(value))
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if hasattr(value, "item"):
        try:
            unwrapped = value.item()
        except (ValueError, TypeError):
            pass
        else:
            if type(unwrapped) is not type(value):
                return _normalize(
                    unwrapped, include_values, evidence_details=evidence_details,
                    _active=_active, _path=_path,
                )
            # Extended floating-point scalars (such as numpy.longdouble) return
            # the same scalar type from .item(). Convert them to Python float
            # and apply the existing non-finite policy: NaN, infinities, and
            # extended values outside IEEE 754 float64 / JSON number range
            # normalize to None.
            if isinstance(unwrapped, Real) and not isinstance(unwrapped, (bool, Integral)):
                try:
                    coerced = float(unwrapped)
                except OverflowError:
                    return None
                return coerced if math.isfinite(coerced) else None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def to_dict(context: Any, *, include_values: bool = False) -> dict[str, Any]:
    """Return an export-safe projection by default; opt in to local sample values.

    Non-finite floats and extended floating-point scalars outside IEEE 754
    float64 / JSON number range are normalized to ``None``.
    """
    result = _normalize(context, include_values)
    if not include_values:
        for column in result.get("columns", []):
            statistics = column["statistics"]
            for field in ("minimum", "maximum", "date_range"):
                statistics[field] = None
            if result["profile"]["row_count"] - statistics["missing_count"] < 3:
                for field in ("mean", "median", "standard_deviation", "skewness"):
                    statistics[field] = None
                statistics["quantiles"] = {}
    return result


def to_json(context: Any, *, indent: int | None = 2, include_values: bool = False) -> str:
    """Serialize a context using the standard JSON encoder."""
    return json.dumps(to_dict(context, include_values=include_values), indent=indent, sort_keys=True, ensure_ascii=False)
