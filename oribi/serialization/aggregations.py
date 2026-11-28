"""Concise qualifications shared by the text exporters."""

from collections.abc import Iterable

from ..models.evidence import Recommendation


def aggregation_conditions(recommendations: Iterable[Recommendation], role: str) -> str:
    conditions = []
    for rec in recommendations:
        balance_snapshot = any(e.code == "name_balance_snapshot_hint" for e in rec.evidence)
        if rec.operation == "SUM" and role == "measure":
            conditions.append("SUM across entities only at a common as-of time" if balance_snapshot else "SUM only for additive quantities")
        elif rec.operation == "MEAN" and role == "measure":
            if balance_snapshot:
                conditions.append("check snapshot intervals and time weighting before averaging")
            elif rec.disposition == "caution":
                conditions.append("check weighting and denominators before averaging")
            else:
                conditions.append("MEAN for roughly symmetric values")
        elif rec.operation == "GROUP BY" and rec.disposition == "caution":
            conditions.append("group only if many small groups are useful")
    return " (" + " / ".join(conditions) + ")" if conditions else ""
