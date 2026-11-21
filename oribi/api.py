"""Public inspection API and pipeline orchestration."""

from dataclasses import dataclass
from numbers import Integral
from typing import cast

import pandas as pd

from .exceptions import InvalidInputError
from .models.context import ColumnContext, DatasetContext
from .models.evidence import CategoricalContext, Warning
from .models.profile import DatasetProfile, IndexContext
from .profiling.cardinality import profile_cardinality
from .profiling.columns import profile_column
from .profiling.observations import AnalysisSession
from .profiling.relationships import detect_relationships, infer_column_relationships
from .profiling.temporal import temporal_gap_warnings
from .reasoning.keys import detect_keys
from .reasoning.rules import dataset_warnings, reason_column
from .reasoning.semantic import infer_semantic


@dataclass(frozen=True)
class ProfileOptions:
    """Controls profiling work while keeping results deterministic.

    ``max_exact_rows`` limits numeric quantiles, composite-key searches, and
    observed relationship scans. Above the limit these calculations are
    skipped. At or below it they still depend on the other profile options
    and on eligible input.
    """

    top_k: int = 10
    include_quantiles: bool = True
    max_exact_rows: int = 100_000
    max_key_columns: int = 8
    max_relationships: int = 40
    max_relationship_columns: int = 40
    max_gap_examples: int = 3
    max_gap_warnings: int = 10

    def __post_init__(self) -> None:
        if type(self.include_quantiles) is not bool:
            raise ValueError("include_quantiles must be a boolean")
        for name in ("top_k", "max_exact_rows", "max_key_columns", "max_relationships",
                     "max_relationship_columns", "max_gap_examples", "max_gap_warnings"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise ValueError(f"{name} must be an integer")
        for name, minimum in (("top_k", 0), ("max_exact_rows", 1), ("max_key_columns", 2),
                              ("max_relationships", 0), ("max_relationship_columns", 0),
                              ("max_gap_examples", 0), ("max_gap_warnings", 1)):
            if getattr(self, name) < minimum:
                raise ValueError(f"{name} must be at least {minimum}")


def _profile_index(df: pd.DataFrame) -> IndexContext | None:
    index = df.index
    default = isinstance(index, pd.RangeIndex) and index.name is None and index.start == 0 and index.step == 1
    if default:
        return None
    names = tuple(index.names)
    dtypes = tuple(str(index.get_level_values(i).dtype) for i in range(index.nlevels))
    if isinstance(index, pd.MultiIndex):
        # Level names may repeat, so use positional labels for this temporary frame.
        index_frame = index.to_frame(index=False, name=list(range(index.nlevels)))
    else:
        index_frame = index.to_frame(index=False)
    missing = int(index_frame.isna().any(axis=1).sum())
    try:
        unique = int(index.nunique()) if index.nlevels == 1 else int(len(index.unique()))
    except (TypeError, ValueError):
        unique = int(len(index.drop_duplicates()))
    # unique_count leaves missing entries out for a single-level index, so
    # len(index) - unique would count each missing entry as a duplicate.
    # index.duplicated() treats repeated missing entries as equal, the same
    # rule pandas uses for is_unique.
    try:
        duplicates = int(index.duplicated().sum())
    except (TypeError, ValueError):
        duplicates = len(index) - unique
    return IndexContext(names, dtypes, index.nlevels, unique, duplicates, missing, bool(index.is_unique))


def inspect(df: pd.DataFrame, *, options: ProfileOptions | None = None) -> DatasetContext:
    """Profile a pandas DataFrame and return its analytical context snapshot.

    The returned context contains no reference to ``df``. Sampled built-in lists
    and dictionaries are copied, including those nested within them. Temporal
    role inference for string or object columns preserves their stored values.
    """
    if options is None:
        options = ProfileOptions()
    if not isinstance(options, ProfileOptions):
        raise InvalidInputError("options must be a ProfileOptions instance")
    session = AnalysisSession(df, options)
    nrows, ncols = df.shape
    exact_row_limit_exceeded = nrows > options.max_exact_rows
    try:
        try:
            duplicates = int(df.duplicated().sum())
        except AttributeError:
            if ncols != 1 or not isinstance(df.dtypes.iloc[0], pd.SparseDtype):
                raise
            # Older pandas uses a Series fast path that cannot compare sparse floats.
            sparse = cast(pd.arrays.SparseArray, df.iloc[:, 0].array)
            duplicates = int(pd.Series(sparse.to_dense()).duplicated().sum())
    except (TypeError, ValueError):
        duplicates = None
    duplicate_rate = (duplicates / nrows) if duplicates is not None and nrows else (0.0 if duplicates == 0 else None)
    names = list(df.columns)
    observations = {name: session.observe(name) for name in names}
    stats = {name: profile_column(obs, nrows, options.top_k, options.include_quantiles and not exact_row_limit_exceeded) for name, obs in observations.items()}
    semantics = {
        name: infer_semantic(
            name,
            observations[name].logical_type,
            stats[name],
            nrows,
            is_float=pd.api.types.is_float_dtype(observations[name].series.dtype),
            date_evidence=observations[name].date_evidence,
        )
        for name in names
    }
    # Unparsed dates can represent the same instant in different forms, and
    # native Period columns represent calendar intervals rather than entity keys.
    structural_roles = {
        name: semantics[name].role for name in names
        if observations[name].date_evidence is None
    }
    key_candidates, grain = detect_keys(
        df, names, stats, structural_roles,
        max_columns=options.max_key_columns, max_exact_rows=options.max_exact_rows,
    )
    cardinalities = {
        name: profile_cardinality(observations[name].series, nrows)
        for name in names
    }
    relationships = detect_relationships(
        df,
        structural_roles,
        max_rows=options.max_exact_rows,
        candidate_keys=key_candidates,
        grain=grain,
        max_relationships=options.max_relationships,
        max_relationship_columns=options.max_relationship_columns,
    )
    column_relationships = infer_column_relationships(
        cardinalities,
        semantics,
        {name: observations[name].logical_type for name in names},
        key_candidates,
        grain,
        relationships,
    )
    columns = []
    for name in names:
        obs = observations[name]
        recommendations, warnings = reason_column(name, obs.logical_type, semantics[name], stats[name], nrows)
        if obs.logical_type == "temporal":
            warnings += temporal_gap_warnings(obs.series, name, max_examples=options.max_gap_examples,
                                              max_warnings=options.max_gap_warnings)
        categories = None
        if (
            semantics[name].role in {"categorical", "dimension"}
            and stats[name].distinct_count > 0
            and stats[name].top_values
        ):
            categories = CategoricalContext(
                values=stats[name].top_values,
                is_complete=len(stats[name].top_values) == stats[name].distinct_count,
        )
        columns.append(ColumnContext(
            name,
            str(obs.series.dtype),
            obs.logical_type,
            stats[name],
            semantics[name],
            recommendations,
            warnings,
            cardinalities[name],
            column_relationships.get(name, ()),
            categories,
        ))
    global_warnings = list(dataset_warnings(duplicates, nrows))
    for col in columns:
        global_warnings.extend(col.warnings)
    if duplicates is None:
        global_warnings.append(Warning(
            "duplicate_check_unavailable", "low", "Duplicate-row check unavailable",
            "Pandas could not compare rows with the available cell values.",
        ))
    return DatasetContext(
        profile=DatasetProfile(
            nrows, ncols, duplicates, duplicate_rate, _profile_index(df),
            exact_row_limit_exceeded=exact_row_limit_exceeded,
        ),
        columns=tuple(columns),
        candidate_keys=key_candidates,
        grain=grain,
        relationships=relationships,
        warnings=tuple(global_warnings),
    )
