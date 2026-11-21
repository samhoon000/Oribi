"""Statistics for observed columns."""

import math
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal
from fractions import Fraction
from heapq import nsmallest
from itertools import groupby
from typing import Any, cast
from zoneinfo import ZoneInfo

import pandas as pd

from ..models.profile import StatisticsContext
from .observations import ColumnObservation


def _snapshot_top_value(value: Any, copies: dict[int, Any]) -> Any:
    """Copy built-in list and dict values without copying arbitrary objects."""
    kind = type(value)
    if kind is not list and kind is not dict:
        return value

    identity = id(value)
    if identity in copies:
        return copies[identity]

    # Register the copy before visiting children so cycles keep their shape.
    if kind is list:
        items: list[Any] = []
        copies[identity] = items
        items.extend(_snapshot_top_value(item, copies) for item in value)
        return items

    entries: dict[Any, Any] = {}
    copies[identity] = entries
    entries.update((key, _snapshot_top_value(item, copies)) for key, item in value.items())
    return entries


# Pandas nullable indexes can yield NumPy scalars instead of Python scalars.
_INTEGER_TYPES = {id(int)} | {
    id(pd.api.types.pandas_dtype(dtype).type)
    for dtype in ("int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64", "longlong", "ulonglong")
}
_PANDAS_SCALARS = tuple(pd.api.types.pandas_dtype(dtype).type for dtype in (
    "bool", "float16", "float32", "float64", "longdouble", "complex64", "complex128",
    "clongdouble", "bytes", "datetime64", "timedelta64",
))
# Key the whitelist by type identity so custom metaclasses cannot influence it.
_SCALAR_TYPES = {
    id(kind): f"{kind.__module__}.{kind.__qualname__}"
    for kind in (bool, float, complex, bytes, Decimal, date, timedelta, pd.Timedelta, pd.Period, *_PANDAS_SCALARS)
}


def _stable_value_key(value: Any) -> tuple[str, Any]:
    """Order known values first. Opaque objects tie in pandas count order."""
    kind = type(value)
    if issubclass(kind, str):
        return "builtins.str", str.__str__(value)
    if id(kind) in _INTEGER_TYPES:
        return "builtins.int", int(value)
    if kind is Fraction:
        return "fractions.Fraction", (value.numerator, value.denominator)
    type_name = _SCALAR_TYPES.get(id(kind))
    if type_name is not None:
        return type_name, repr(value)
    if kind is pd.Timestamp:
        return "pandas.Timestamp", str(value.asm8)
    if kind is pd.Interval:
        return "pandas.Interval", (_stable_value_key(value.left), _stable_value_key(value.right), value.closed)
    if kind is datetime or kind is time:
        tz_type = type(value.tzinfo)
        if value.tzinfo is None or tz_type is timezone or tz_type is ZoneInfo:
            return f"datetime.{kind.__name__}", value.isoformat()
    if kind is tuple or kind is frozenset:
        keys = tuple(_stable_value_key(item) for item in value)
        return f"builtins.{kind.__name__}", tuple(sorted(keys)) if kind is frozenset else keys
    return "~", ""


def _top_values(counts: pd.Series, limit: int) -> tuple[tuple[Any, int], ...]:
    """Select by frequency first and rank only ties that can affect the result."""
    if len(counts) <= max(32, limit):
        ordered = sorted(
            ((value, int(count)) for value, count in counts.items() if count > 0),
            key=lambda item: (-item[1], _stable_value_key(item[0])),
        )
        return tuple(ordered[:limit])

    # There are usually few distinct frequencies even for very wide ties.
    frequencies = counts.value_counts(sort=False).sort_index(ascending=False)
    cutoff = frequencies.index[int(frequencies.cumsum().searchsorted(limit))]
    if cutoff == 0:
        return _top_values(counts[counts > 0], limit)

    ranked: list[tuple[Any, int]] = []
    ties = counts.index
    if len(frequencies) > 1:
        leaders = counts[counts > cutoff].sort_values(ascending=False, kind="stable")
        for count, group in groupby(leaders.items(), key=lambda item: item[1]):
            values = [value for value, _ in group]
            if len(values) > 1:
                values.sort(key=_stable_value_key)
            ranked.extend((value, int(count)) for value in values)
        ties = counts.index[counts == cutoff]
    value_index = ties.categories if isinstance(ties, pd.CategoricalIndex) else ties
    key = str.__str__ if value_index.inferred_type == "string" else _stable_value_key
    ranked.extend((value, int(cutoff)) for value in nsmallest(limit - len(ranked), ties, key=key))
    return tuple(ranked)


def profile_column(
    obs: ColumnObservation, row_count: int, top_k: int, include_quantiles: bool,
) -> tuple[StatisticsContext, bool]:
    """Return public statistics and the observed fractional-value signal."""
    s = obs.series
    denom = row_count or 1
    non_null = s.dropna()
    values: dict[str, Any] = {}
    has_fractional_values = False
    if obs.logical_type == "numeric" and len(non_null):
        numeric_series = (
            cast(Any, non_null).sparse.to_dense()
            if isinstance(non_null.dtype, pd.SparseDtype)
            else non_null
        )
        numeric = pd.to_numeric(numeric_series, errors="coerce").dropna()
        if len(numeric):
            def finite_or_none(value: Any) -> float | None:
                if pd.isna(value):
                    return None
                result = float(value)
                return result if math.isfinite(result) else None

            # Pandas' mean/median/std/skew emit a RuntimeWarning and can yield
            # NaN when the series holds infinities (e.g. nanops computes
            # (avg - values) ** 2 over inf). Compute the moment statistics on
            # the finite subset so no warning is raised; infinities are simply
            # excluded rather than turning the whole statistic into NaN.
            finite = numeric[numeric.map(math.isfinite)]
            if pd.api.types.is_float_dtype(finite.dtype):
                # Use the already-filtered finite values; Arrow-backed Series
                # do not implement modulo.
                has_fractional_values = bool(((finite.to_numpy() % 1) != 0).any())
            values.update(
                minimum=finite.min() if len(finite) else None,
                maximum=finite.max() if len(finite) else None,
                mean=finite_or_none(finite.mean()) if len(finite) else None,
                median=finite_or_none(finite.median()) if len(finite) else None,
                standard_deviation=finite_or_none(finite.std()) if len(finite) > 1 else None,
            )
            values["zero_count"] = int((numeric == 0).sum())
            values["negative_count"] = int((numeric < 0).sum())
            if include_quantiles:
                values["quantiles"] = (
                    {str(q): finite_or_none(v) for q, v in finite.quantile([.25, .5, .75]).items()}
                    if len(finite) else {}
                )
            if len(finite) >= 3 and finite.nunique() > 1:
                skew_raw: Any = finite.skew()
                skew = float(skew_raw)
                values["skewness"] = skew if pd.notna(skew) else None
    elif obs.logical_type == "temporal" and len(non_null):
        values["minimum"] = non_null.min()
        values["maximum"] = non_null.max()
        values["date_range"] = f"{values['minimum']} to {values['maximum']}"
    if top_k > 0 and len(non_null) and (obs.logical_type in {"text", "categorical", "boolean", "unknown", "duration"} or obs.distinct_count <= top_k * 4):
        try:
            counts = non_null.value_counts(sort=False, dropna=True)
        except (TypeError, ValueError):
            pass
        else:
            ranked = _top_values(counts, top_k)
            values["top_values"] = tuple((_snapshot_top_value(k, {}), count) for k, count in ranked)
            values["top_frequencies"] = tuple(count for _, count in ranked)
    statistics = StatisticsContext(
        missing_count=obs.missing_count,
        missing_rate=obs.missing_count / denom,
        distinct_count=obs.distinct_count,
        distinct_rate=obs.distinct_count / denom,
        **values,
    )
    return statistics, has_fractional_values
