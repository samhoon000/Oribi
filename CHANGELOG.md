# Changelog

All notable changes to Oribi are recorded here.

## Unreleased

### Added and improved

- Explicitly flag unsafe operations (such as SUM and MEAN) as inappropriate for identifier values.
- Explain why counting rows differs from counting distinct entities when identifiers repeat or duplicate rows occur.
- Add aggregation recommendations and usage guidance for categorical and dimension columns, explaining when grouping or counting distinct values is useful.
- Provide explicit starting-point rationale for SUM and MEAN on numeric measures while preserving cautions for negative values and skew.
- Surface inappropriate aggregation guidance and useful operations (such as COUNT DISTINCT on categorical columns or MEDIAN on measures) in Markdown and compact context reports.

### Fixed

- Detach mutable list and dictionary values nested inside tuple samples when snapshotting top values, so editing the source data after inspection no longer changes an already-captured context.

## 0.2.0 — 2026-11-19

### Compatibility

- Dictionary and JSON exports now use schema version `2`. Exports omit sampled cell values and exact bounds by default. Use `include_values=True` for local samples and bounds. Dictionary and JSON exports retain frequency counts. Consumers that parse the previous Markdown or compact context layout should update their parsers.
- `ProfileOptions` validates booleans and integer limits when constructed. Native complex-valued columns raise `InvalidInputError`.

### Added and improved

- Report per-column cardinality, relationships to the observation grain, and one-to-one and one-to-many value mappings. Distinguish exact dependencies from approximate mappings with contradiction and paired-row evidence. Exclude numeric measures from entity grouping inference.
- Report verified unique keys separately from approximate candidates. Colliding or null-bearing key candidates cannot establish grain. Improve role inference for repeated low-diversity codes and numeric columns.
- Profile meaningful pandas indexes separately and improve the Markdown and compact context reports. Compact output retains schema information on wide frames.
- Bound relationship discovery and temporal-gap warnings and examples. Document that the `max_exact_rows` limit includes the boundary for quantiles, composite keys, and observed relationships.
- Warn when a text column has more than 100 distinct values and is not inferred as an identifier.

### Fixed

- Count duplicate indexes consistently with pandas and handle non-string index names and repeated MultiIndex level names.
- Profile real numeric object columns with missing values, exclude infinities from summary moments and quantiles, and preserve exact integer bounds. Keep positional construction of `StatisticsContext` compatible with 0.1.0.
- Exclude unused categories from top values and detach sampled mutable lists and dictionaries from source cells.
- Classify elapsed durations as `duration` and exclude them from single-column key selection. Preserve nanosecond cadences and timezone offsets in temporal output.
- Escape Markdown in column names and opt-in value samples. Show each full analytical warning once.

### Documentation and packaging

- Refresh the README and contributor file map. Validate the wheel and source distribution, test the installed wheel on Python 3.10–3.14, and publish through GitHub Actions when a GitHub Release is published.

## 0.1.0 — Initial release

- Introduce the `oribi` pandas DataFrame profiling API.
- Report column statistics, evidence-backed semantic roles, recommendations, warnings, relationships, candidate keys, and likely grain.
- Add conservative temporal-gap warnings for regular datetime series.
- Provide summary, dictionary, JSON, Markdown, and bounded AI-oriented context output.
