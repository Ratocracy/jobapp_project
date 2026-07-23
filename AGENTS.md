# AGENTS.md

## Project purpose

This repository builds an ETL pipeline for job-posting and resume-screening
data. Its outputs will support candidate-job retrieval, ranking, and screening
models.

The current priority is reliable, reproducible data engineering. Do not add
complex ML models until the cleaned datasets, validation rules, and leakage-safe
splits are established.

## Scope

- Treat files under `raw_data/` as immutable source data.
- Never edit, rename, delete, or overwrite raw files.
- Write generated data only to designated output directories.
- Do not commit generated datasets, credentials, or large binary artifacts.
- Do not access external systems or S3 unless the task explicitly requires it.
- Do not change AWS configuration or credentials.

## Development workflow

- Inspect existing code and schemas before making changes.
- State assumptions when requirements are ambiguous.
- Prefer small, reviewable changes.
- Do not refactor unrelated code.
- Before adding a dependency, explain why the existing dependencies are
  insufficient.
- Do not execute expensive full-dataset jobs until logic has been tested on a
  deterministic sample.
- Obtain approval before operations that may incur cloud costs.

## Pipeline architecture

Use these logical layers:

1. Bronze: source-preserving ingestion.
2. Silver: typed, normalized, deduplicated data.
3. Gold: model-ready datasets and deterministic split assignments.

Keep ingestion, transformation, validation, and model-feature generation
separate.

## Data rules

- Use `job_link` as the source key for job-posting joins.
- Preserve raw columns when creating normalized versions.
- Use explicit schemas instead of relying on schema inference in production.
- Convert string booleans and timestamps to proper types.
- Retain incomplete job records and attach quality flags instead of silently
  dropping them.
- Record row counts, rejected records, null rates, duplicate rates, and join
  coverage for each run.
- Make transformations deterministic and idempotent.
- Document every rule that removes or materially changes records.

## ML and leakage guardrails

- Never use `Reason_for_decision` as a model feature.
- Do not use post-outcome or operational fields as predictive features.
- Do not randomly split resume-screening rows.
- Group splits by normalized job-description identity or template.
- Fit vocabularies, imputers, encoders, and other learned transformations on
  training data only.
- Keep a locked test set that is not used during feature development.
- Mask names, email addresses, phone numbers, and other direct identifiers
  before modeling.
- Evaluate performance by role and other relevant subgroups, not only globally.
- Treat job recommendations and screening scores as decision support, not
  autonomous hiring decisions.

## PySpark conventions

- Prefer Spark transformations for production-scale processing.
- Avoid collecting full datasets to the driver.
- Avoid Python UDFs when Spark SQL functions can implement the transformation.
- Avoid unnecessary actions such as repeated `count()` calls.
- Do not use `repartition(1)` on large outputs.
- Use Parquet for intermediate and final tabular datasets.
- Partition outputs only when the partition field supports common access
  patterns and has reasonable cardinality.

## Notebook conventions

- Use notebooks for exploration and pipeline demonstration, not as the sole
  production implementation.
- Move reusable transformations into importable Python modules.
- Keep notebook cells restartable and executable from top to bottom.
- Do not embed machine-specific paths, usernames, credentials, or secrets.
- Parameterize local and S3 input/output locations.

## Testing requirements

For each material transformation, test:

- Expected schema and data types.
- Primary-key uniqueness.
- Null and allowed-value rules.
- Join cardinality.
- Deduplication behavior.
- Deterministic output.
- Leakage-free split assignment.
- Behavior on malformed and missing inputs.

Start with small fixtures or deterministic samples before running against the
complete dataset.

## Verification before completion

Before declaring work complete:

- Run the relevant tests.
- Report commands executed and their results.
- Report any checks that could not be run.
- Summarize input/output row counts for pipeline changes.
- Confirm that raw data was not modified.
- Confirm that no secrets or generated datasets were added to version control.