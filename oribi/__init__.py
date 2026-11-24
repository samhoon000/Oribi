"""Build structured analytical context from pandas DataFrames."""

from .api import ProfileOptions, inspect
from .exceptions import InvalidInputError, OribiError
from .models import (
    CardinalityContext,
    ColumnContext,
    ColumnRelationshipContext,
    DatasetContext,
    Evidence,
    GrainContext,
    KeyCandidate,
    MetricDefinition,
    Recommendation,
    RelationshipContext,
    SemanticContext,
    StatisticsContext,
    Warning,
)

__all__ = [
    "CardinalityContext", "ColumnContext", "ColumnRelationshipContext", "DatasetContext", "Evidence", "GrainContext", "InvalidInputError",
    "KeyCandidate", "MetricDefinition", "ProfileOptions", "Recommendation", "RelationshipContext",
    "OribiError", "SemanticContext", "StatisticsContext", "Warning", "inspect",
]

__version__ = "0.3.0"
