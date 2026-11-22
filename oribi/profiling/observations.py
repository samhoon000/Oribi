"""Per-call memoized observations shared across detectors."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import pandas as pd

from ..exceptions import InvalidInputError
from ..models.evidence import Evidence
from .temporal import date_like_evidence

if TYPE_CHECKING:
    from ..api import ProfileOptions


@dataclass
class ColumnObservation:
    name: str
    series: pd.Series
    missing_count: int
    distinct_count: int
    logical_type: str
    stats: dict[str, Any] = field(default_factory=dict)
    date_evidence: Evidence | None = None


class AnalysisSession:
    def __init__(self, frame: pd.DataFrame, options: ProfileOptions):
        if not isinstance(frame, pd.DataFrame):
            raise InvalidInputError("inspect() expects a pandas DataFrame")
        if not frame.columns.is_unique:
            raise InvalidInputError("DataFrame column labels must be unique")
        if any(not isinstance(c, str) for c in frame.columns):
            raise InvalidInputError("DataFrame column labels must be strings")
        self.frame = frame
        self.options = options
        self.observations: dict[Any, ColumnObservation] = {}

    def observe(self, name: Any) -> ColumnObservation:
        if name in self.observations:
            return self.observations[name]
        series = self.frame[name]
        if pd.api.types.is_complex_dtype(series.dtype):
            raise InvalidInputError(f"Complex-valued column {name!r} is not supported")
        missing = int(series.isna().sum())
        try:
            distinct = int(series.nunique(dropna=True))
        except (TypeError, ValueError):
            distinct = int(series.dropna().astype(str).nunique())
        date_evidence = None
        if pd.api.types.is_bool_dtype(series.dtype):
            logical = "boolean"
        elif pd.api.types.is_datetime64_any_dtype(series.dtype):
            logical = "temporal"
        elif pd.api.types.is_numeric_dtype(series.dtype):
            logical = "numeric"
        elif getattr(series.dtype, "kind", None) == "m":
            # Elapsed durations (timedelta64) are neither identifiers nor measures here.
            logical = "duration"
        elif isinstance(series.dtype, pd.CategoricalDtype):
            logical = "categorical"
        elif isinstance(series.dtype, pd.PeriodDtype):
            # Native Period columns represent calendar intervals rather than
            # point timestamps or entity identifiers.
            logical = "unknown"
            date_evidence = Evidence(
                "period_dtype",
                "Column has a pandas Period dtype",
                {"dtype": str(series.dtype)},
            )
        elif series.dtype == object or pd.api.types.is_string_dtype(series.dtype):
            values = series.dropna()
            kind = "empty" if values.empty else pd.api.types.infer_dtype(values, skipna=False)
            if kind == "empty":
                logical = "unknown"
            elif kind in {"integer", "floating", "mixed-integer-float"}:
                logical = "numeric"
            elif kind == "string":
                logical = "text"
            else:
                first_type = type(values.iloc[0])
                logical = "text" if all(type(value) is first_type for value in values) else "unknown"
            if logical in {"text", "unknown"}:
                date_evidence = date_like_evidence(values, kind, distinct)
        else:
            logical = "unknown"
        obs = ColumnObservation(str(name), series, missing, distinct, logical, date_evidence=date_evidence)
        self.observations[name] = obs
        return obs
