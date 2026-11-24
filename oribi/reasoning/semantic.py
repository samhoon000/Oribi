"""Composable rules for likely column roles."""

import re

from ..models.evidence import Evidence, SemanticContext
from ..models.profile import StatisticsContext

_MIN_ROLE_SCORE = 0.55
_AMBIGUITY_MARGIN = 0.12
_MEASURE_NAME_TOKENS = {
    "amount", "balance", "charge", "cost", "discount", "fee",
    "margin", "price", "profit", "qty", "quantity", "revenue",
    "salary", "sales", "spend", "subtotal", "tax", "total",
}
_LABEL_NAME_TOKENS = {
    "band", "bands", "bin", "bins", "bucket", "buckets", "category", "categories",
    "class", "classes", "code", "codes", "label", "labels", "tier", "tiers",
}
_SUBJECT_NAME_TOKENS = _MEASURE_NAME_TOKENS | _LABEL_NAME_TOKENS
_RANK_NAME_TOKENS = {"rank", "ranks", "ranking", "percentile", "percentiles", "quantile", "quantiles", "decile", "deciles"}
_SOFT_IDENTIFIER_SUFFIX_TOKENS = {"no", "num", "number"}
_CAMEL_BOUNDARY_UPPER = re.compile(r"([A-Z]+)([A-Z][a-z])")
_CAMEL_BOUNDARY_LOWER = re.compile(r"([a-z0-9])([A-Z])")
_NAME_WORD = re.compile(r"[a-z0-9]+")
_VERSION_SUFFIX = re.compile(r"(?<=[^a-z0-9])v(?:er(?:sion)?)?[^a-z0-9]*[0-9]+$")


def column_name_words(name: str) -> list[str]:
    segmented = _CAMEL_BOUNDARY_UPPER.sub(r"\1 \2", str(name))
    segmented = _CAMEL_BOUNDARY_LOWER.sub(r"\1 \2", segmented)
    normalized = segmented.lower()
    if normalized[-1:].isdigit():
        normalized = _VERSION_SUFFIX.sub("", normalized)
    return _NAME_WORD.findall(normalized)


def infer_semantic(
    name: str,
    logical_type: str,
    stats: StatisticsContext,
    row_count: int,
    *,
    has_fractional_values: bool,
    is_bool_dtype: bool,
    is_decimal: bool = False,
    is_float: bool = False,
    date_evidence: Evidence | None = None,
) -> SemanticContext:
    # Decimal values support the same measure heuristics as native floats.
    is_float = is_float or is_decimal
    votes: dict[str, float] = {}
    evidence: list[Evidence] = []
    distinct = stats.distinct_count
    rate = stats.distinct_rate
    negative_count = getattr(stats, "negative_count", 0) or 0
    std = getattr(stats, "standard_deviation", None)
    has_dispersion = std is not None and std > 0
    words = column_name_words(name)
    tokens = set(words)
    explicit_identifier_name = "id" in tokens or (len(words) >= 2 and words[-1] == "key")
    code_name = len(words) >= 2 and words[-1] == "code"
    soft_identifier_name = len(words) >= 2 and words[-1] in _SOFT_IDENTIFIER_SUFFIX_TOKENS
    # Label a measure by its subject, before any grouping qualifier such as
    # "by category". A leading qualifier in "category revenue" can describe the measure.
    subject = words[:words.index("by")] if "by" in tokens else words
    subject_kind = None
    for word in reversed(subject):
        if word in _SUBJECT_NAME_TOKENS:
            subject_kind = word
            break
    label_name = subject_kind in _LABEL_NAME_TOKENS or not _RANK_NAME_TOKENS.isdisjoint(subject)
    measure_name = (
        not (label_name or explicit_identifier_name or code_name or soft_identifier_name)
        and subject_kind in _MEASURE_NAME_TOKENS
    )
    observed = row_count - stats.missing_count
    unique_named_measure = measure_name and distinct > 1 and distinct == observed
    # A suffix alone cannot establish identity when only a few codes recur.
    categorical_code_shape = row_count >= 20 and distinct <= 3 and rate <= 0.15
    if explicit_identifier_name and distinct > 1:
        votes["identifier"] = .48 if categorical_code_shape else .78
        evidence.append(Evidence("name_identifier_hint", "Column name strongly suggests an identifier"))
        if categorical_code_shape:
            evidence.append(Evidence("few_repeated_codes", "A few values repeat across many rows; entity identity is unestablished", {"distinct_count": distinct, "row_count": row_count}))
        evidence.append(Evidence(
            "identifier_value_variation",
            f"Observed {distinct} distinct non-missing values",
            {"distinct_count": distinct, "row_count": row_count},
        ))
    elif code_name and distinct > 1:
        votes["identifier"] = .48 if categorical_code_shape else .75
        evidence.append(Evidence("name_identifier_hint", "Column name suggests an identifier"))
        evidence.append(Evidence(
            "identifier_value_variation",
            f"Observed {distinct} distinct non-missing values",
            {"distinct_count": distinct, "row_count": row_count},
        ))
    elif soft_identifier_name and distinct > 1:
        votes["identifier"] = .46 if categorical_code_shape else .68
        evidence.append(Evidence("name_identifier_hint", "Column name suggests an identifier"))
        evidence.append(Evidence(
            "identifier_value_variation",
            f"Observed {distinct} distinct non-missing values",
            {"distinct_count": distinct, "row_count": row_count},
        ))

    if row_count and rate >= .98 and distinct > 1 and logical_type != "duration":
        # Uniqueness is supporting evidence, not proof of business identity.
        # Elapsed durations are skipped: unique durations do not identify entities.
        # It is deliberately weaker for numeric columns, which commonly hold
        # continuous measures that happen to be unique in small samples.
        unique_vote = .28 if logical_type == "numeric" else .62
        if logical_type == "numeric" and not is_float and negative_count == 0 and soft_identifier_name:
            unique_vote = .75
        votes["identifier"] = max(votes.get("identifier", 0), unique_vote)
        evidence.append(Evidence("mostly_unique", f"{distinct} distinct non-null values across {row_count} rows", {"distinct_count": distinct, "row_count": row_count, "distinct_rate": rate}))
    elif (
        logical_type == "numeric"
        and not is_float
        and negative_count == 0
        and not (measure_name or label_name)
        and row_count >= 10
        and .90 <= rate < .98
    ):
        votes["identifier"] = max(votes.get("identifier", 0), .50)
        evidence.append(Evidence(
            "high_integer_uniqueness",
            f"{distinct} distinct non-negative integer values across {row_count} rows",
            {"distinct_count": distinct, "row_count": row_count, "distinct_rate": rate},
        ))

    is_low_cardinality_numeric = (
        logical_type == "numeric"
        and (not is_float or label_name)
        and not unique_named_measure
        and (
            distinct <= min(20, max(3, row_count // 10))
            or (row_count >= 20 and rate <= .1)
        )
    )

    if logical_type == "numeric":
        measure_score = .48
        if is_decimal:
            evidence.append(Evidence("decimal_values", "Observed non-missing values are Decimal numbers"))
        else:
            evidence.append(Evidence("numeric_type", "Column has a numeric dtype"))
        if measure_name and not is_low_cardinality_numeric:
            measure_score += .08
            evidence.append(Evidence("name_measure_hint", "Column name supports a quantitative measure"))
        if is_float and not label_name and not explicit_identifier_name:
            measure_score += .08
            # Decimal values may be stored as integers by the SQL engine.
            if not is_decimal:
                evidence.append(Evidence("floating_point_type", "Floating-point values are consistent with a quantitative measure"))
        elif not label_name and not is_low_cardinality_numeric and row_count and rate < .98 and distinct > min(20, max(3, row_count // 10)):
            measure_score += .08
            evidence.append(Evidence("repeated_numeric_measure_values", "Numeric values repeat across rows without having low categorical cardinality"))
        if negative_count > 0 and not label_name and not is_low_cardinality_numeric:
            measure_score += .06
            evidence.append(Evidence("negative_numeric_values", "Negative numeric values are consistent with signed measurements", {"negative_count": negative_count}))
        if has_dispersion and not label_name and not is_low_cardinality_numeric and not explicit_identifier_name and (measure_name or is_float or rate < .98):
            measure_score += .06
            evidence.append(Evidence("numeric_value_variability", "Numeric values exhibit non-zero dispersion", {"standard_deviation": std}))
        if distinct <= 1 or std == 0:
            # Constant numeric columns lack variability to serve as active measures or identifiers.
            measure_score = min(measure_score, .48)
            if "identifier" in votes:
                votes["identifier"] = min(votes["identifier"], .48)
            if std == 0:
                evidence.append(Evidence("constant_numeric_values", "Numeric values have zero dispersion"))
        if "identifier" in votes and (negative_count > 0 or has_fractional_values or (is_float and not explicit_identifier_name)):
            # Negative or fractional values conflict with entity identifier semantics.
            votes["identifier"] = round(votes["identifier"] * .60, 2)
            evidence.append(Evidence("non_identifier_numeric_domain", "Floating-point or negative values conflict with entity identity"))
        votes["measure"] = min(measure_score, .92)
    elif logical_type == "temporal":
        votes["temporal"] = .86
        evidence.append(Evidence("datetime_dtype", "Column has a datetime dtype"))
    elif date_evidence is not None:
        votes["temporal"] = .84
        evidence.append(date_evidence)
    elif logical_type == "boolean":
        votes["boolean"] = .9
        if is_bool_dtype:
            evidence.append(Evidence("boolean_dtype", "Column has a boolean dtype"))
        else:
            evidence.append(Evidence("boolean_values", "Observed non-missing values are boolean"))
    elif logical_type == "categorical":
        votes["categorical"] = .78
        evidence.append(Evidence("categorical_dtype", "Column dtype supports categorical grouping"))
    elif logical_type == "text":
        if rate > .5 and distinct > 20:
            votes["text"] = .62
            evidence.append(Evidence("high_cardinality_text", "Text values have relatively high cardinality"))
        else:
            votes["dimension"] = .62
            evidence.append(Evidence("text_dimension", "Text values are reusable across rows"))
    if is_low_cardinality_numeric:
        votes["dimension"] = max(votes.get("dimension", 0), .72 if row_count >= 20 and rate <= .1 else .64)
        evidence.append(Evidence("low_cardinality_numeric", "Numeric values have low cardinality"))
    if not votes:
        return SemanticContext("unknown", 0.0, tuple(evidence))
    role, score = max(votes.items(), key=lambda item: (item[1], item[0]))
    ranked_votes = sorted(votes.items(), key=lambda item: (-item[1], item[0]))
    if score < _MIN_ROLE_SCORE:
        return SemanticContext("unknown", round(score, 2), tuple(evidence))
    if len(ranked_votes) > 1 and score - ranked_votes[1][1] < _AMBIGUITY_MARGIN:
        return SemanticContext("unknown", round(min(score, .5), 2), tuple(evidence))
    return SemanticContext(role, round(min(score, .98), 2), tuple(evidence))
