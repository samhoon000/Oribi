"""Structured warnings and aggregation recommendations."""

from ..models.evidence import Evidence, Recommendation, SemanticContext, Warning
from ..models.profile import StatisticsContext
from .semantic import column_name_words

_NON_ADDITIVE_TOKENS = {
    "average", "avg", "cagr", "ctr", "median", "per", "pct", "percent", "percentage", "percentages",
    "proportion", "proportions", "rate", "rates", "ratio", "ratios", "roi", "share", "shares",
}
_NON_ADDITIVE_TOKEN_PAIRS = {("unit", "price"), ("unit", "cost")}


def reason_column(name: str, logical_type: str, semantic: SemanticContext, stats: StatisticsContext, row_count: int) -> tuple[tuple[Recommendation, ...], tuple[Warning, ...]]:
    """Build guidance, treating adjacent unit price and unit cost name tokens as non-additive hints."""
    recommendations: list[Recommendation] = []
    warnings: list[Warning] = []
    high_cardinality_text = logical_type == "text" and semantic.role not in {"identifier", "temporal"} and stats.distinct_count > 100
    high_cardinality_category = logical_type == "categorical" and stats.distinct_count > 100
    if stats.missing_rate >= .2:
        warnings.append(Warning("high_missingness", "medium", "High missingness", f"{name} is missing {stats.missing_rate:.1%} of its values.", (Evidence("missing_rate", f"{stats.missing_count} of values are missing", {"missing_count": stats.missing_count, "missing_rate": stats.missing_rate}),), "Analyses using this column may exclude many rows.", "Check whether missing values have a meaningful interpretation."))
    if stats.distinct_count == 0:
        return (), tuple(warnings)
    if semantic.role == "identifier":
        is_repeated = stats.distinct_count < row_count - stats.missing_count
        if is_repeated:
            cd_rationale = "Distinct counting is recommended to count unique entities because repeated identifier values cause row counts to exceed entity counts."
        else:
            cd_rationale = "Distinct counting is recommended for counting unique entities rather than summing or averaging identifier values."
        recommendations.append(Recommendation("COUNT DISTINCT", "recommended", cd_rationale, semantic.evidence))
        recommendations.append(Recommendation("SUM", "not_applicable", "Summing identifier values is inappropriate because identifiers represent entity labels rather than additive quantities.", semantic.evidence))
        recommendations.append(Recommendation("MEAN", "not_applicable", "Averaging identifier values is inappropriate because identifiers represent entity labels rather than measurable quantities.", semantic.evidence))
        if is_repeated:
            warnings.append(Warning("non_unique_identifier", "medium", "Identifier-like column is not unique", f"{name} has {stats.distinct_count} distinct values across rows and may identify repeated entities.", semantic.evidence, "Row counts measure record occurrences rather than unique entities; when identifiers repeat across rows, counting rows overcounts entities.", "Use COUNT DISTINCT when counting unique entities, as row counts and entity counts represent different measurements."))
    elif semantic.role == "measure":
        words = column_name_words(name)
        if (
            "%" in name
            or not _NON_ADDITIVE_TOKENS.isdisjoint(words)
            or not _NON_ADDITIVE_TOKEN_PAIRS.isdisjoint(zip(words, words[1:], strict=False))
        ):
            evidence = (*semantic.evidence, Evidence("name_non_additive_hint", "The name suggests a rate, ratio, or other derived measure"))
            recommendations.append(Recommendation("SUM", "not_applicable", "Summing rates, ratios, or pre-aggregated measures across observations generally does not produce a meaningful total.", evidence))
            recommendations.append(Recommendation("MEAN", "caution", "Check weighting and denominators before combining derived values. Recompute a combined rate or ratio from its underlying totals when available.", evidence))
        else:
            recommendations.append(Recommendation("SUM", "recommended", "SUM is a reasonable starting point for additive quantities to measure total volume across observations.", semantic.evidence))
            recommendations.append(Recommendation("MEAN", "recommended", "MEAN is a reasonable starting point for estimating typical values across observations when the distribution is relatively symmetric.", semantic.evidence))
        recommendations.append(Recommendation("MEDIAN", "useful", "MEDIAN is useful for central tendency, especially when the distribution may be skewed or contain extreme values.", semantic.evidence))
        recommendations.append(Recommendation("MIN", "valid", "MIN is valid for observing lower bounds across observations.", semantic.evidence))
        recommendations.append(Recommendation("MAX", "valid", "MAX is valid for observing upper bounds across observations.", semantic.evidence))
        if stats.skewness is not None and abs(stats.skewness) >= 2:
            tail = "small values" if stats.skewness < 0 else "large values"
            warnings.append(Warning(
                "high_skew",
                "low",
                "Highly skewed measure",
                f"{name} has skewness {stats.skewness:.2f}.",
                (Evidence("skewness", "Observed distribution is strongly skewed", {"skewness": stats.skewness}),),
                f"The mean may be sensitive to a small number of {tail}.",
                "Consider reporting the median alongside the mean.",
            ))
        if stats.negative_count:
            warnings.append(Warning("negative_measure_values", "low", f"Negative values in {name}", f"{stats.negative_count} negative values were observed in likely measure {name}.", (Evidence("negative_count", f"{stats.negative_count} values are below zero", {"negative_count": stats.negative_count}),), "Negative values may represent refunds, reversals, or data issues.", "Check domain meaning before filtering negative values."))
    elif semantic.role == "dimension" and not high_cardinality_text:
        recommendations.append(Recommendation(
            "GROUP BY",
            "recommended",
            "Grouping by this dimension is useful for segmenting aggregations across categories, and counting distinct values is useful for assessing category cardinality.",
            semantic.evidence,
        ))
    elif semantic.role == "categorical":
        recommendations.append(Recommendation(
            "GROUP BY",
            "caution" if high_cardinality_category else "recommended",
            "Group only if many small groups are useful for the analysis."
            if high_cardinality_category else "Grouping is useful for segmenting aggregations across categories.",
            semantic.evidence,
        ))
        recommendations.append(Recommendation(
            "COUNT DISTINCT",
            "useful",
            "Counting distinct values is useful for measuring category cardinality or diversity.",
            semantic.evidence,
        ))
    if high_cardinality_category:
        warnings.append(Warning("high_cardinality_category", "low", "High-cardinality category", f"{name} has {stats.distinct_count} distinct categories.", (), "Grouping by this column may produce many small groups."))
    if high_cardinality_text:
        # Identifier-like text is expected to be near-unique; it has its own
        # non_unique_identifier handling instead.
        observed = row_count - stats.missing_count
        ratio = stats.distinct_count / observed if observed else 0.0
        warnings.append(Warning("high_cardinality_text", "low", "High-cardinality text", f"{name} has {stats.distinct_count} distinct text values across {observed} non-missing rows.", (Evidence("distinct_count", f"{stats.distinct_count} distinct values ({ratio:.1%} of non-missing rows)", {"distinct_count": stats.distinct_count, "non_missing_count": observed, "uniqueness_ratio": ratio}),), "Grouping or aggregating by this column may produce many small groups.", "Use it as a label or free text rather than as a grouping dimension, or bucket it before grouping."))
    return tuple(recommendations), tuple(warnings)


def dataset_warnings(duplicate_count: int | None, row_count: int) -> tuple[Warning, ...]:
    if duplicate_count:
        return (Warning("duplicate_rows", "medium", "Duplicate rows detected", f"{duplicate_count} fully duplicated rows were observed.", (Evidence("duplicate_count", f"{duplicate_count} duplicate rows across {row_count} total rows", {"duplicate_row_count": duplicate_count, "row_count": row_count}),), "Row counts and entity counts are different measurements; duplicates can inflate aggregations, though repeated rows may be valid depending on data grain.", "Confirm whether repeated rows reflect valid repeated events or redundant data before aggregation."),)
    return ()
