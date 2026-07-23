# Job Application Retrieval Pipeline

This project builds a distributed ETL and NLP pipeline for matching resumes to
job postings. The current MVP is an unsupervised information-retrieval system:
given a resume, it retrieves and ranks similar jobs from a LinkedIn job catalog.

The system is decision support and must not be used to make autonomous hiring
decisions.

## Current status

The Week 1-3 MVP is implemented and runs end to end:

1. Read raw Parquet sources with explicit PySpark schemas.
2. Create normalized jobs and resumes silver tables.
3. Convert jobs and resumes into a shared document contract.
4. Fit TF-IDF corpus statistics on job documents.
5. Transform jobs and resumes into the same normalized vector space.
6. Generate approximate candidates with LSH.
7. Rerank candidates using exact cosine similarity.
8. Write top-K recommendations, fitted models, evaluation scores, and timing
   metrics.

The current model is only the classical NLP baseline:

```text
RegexTokenizer
-> StopWordsRemover
-> HashingTF
-> IDF fitted on jobs
-> L2 normalization
-> BucketedRandomProjectionLSH
-> exact cosine reranking
-> top-K jobs per resume
```

There is currently no neural network, transformer encoder, cross-encoder, or
supervised learning-to-rank model.

## Retrieval architecture

```text
Job title + summary + skills
              |
              v
        job combined_text --------+
                                  |
                                  v
                         shared fitted TF-IDF
                                  |
                                  v
PII-masked resume + role -> resume combined_text
                                  |
                                  v
                         normalized vectors
                                  |
                                  v
                        LSH candidate search
                                  |
                                  v
                       exact cosine reranking
                                  |
                                  v
                         top-K recommendations
```

The paired resume `Job_Description`, `Decision`, and `Reason_for_decision` are
excluded from retrieval inputs. They are retained only for auditing and proxy
evaluation.

## Data layers

The downloaded source Parquets act as the landing/bronze inputs. They are
immutable and are not copied into a second bronze directory for the MVP.

Silver outputs:

- `data/silver/jobs_sample`: one enriched row per sampled job posting.
- `data/silver/resumes`: cleaned resumes with stable IDs, masked direct PII,
  labels, and job-description-grouped splits.

Gold outputs:

- `data/gold/job_features`: tokenized job documents and TF-IDF vectors.
- `data/gold/resume_features`: resume query documents in the same vector space.
- `data/gold/top_k_recommendations`: ranked resume-job matches.
- `data/gold/evaluation_scores`: labeled resume/job-description proxy scores.
- `data/gold/models/tfidf`: fitted Spark feature pipeline.
- `data/gold/models/lsh`: fitted, seeded LSH model.
- `data/gold/benchmark_metrics.json`: scale, coverage, quality, and runtime
  metrics from the latest run.

Raw and generated datasets are excluded from Git.

## Latest benchmark

The latest local run used a deterministic 1% job sample and 100 resume queries:

| Metric | Result |
|---|---:|
| Job documents | 13,634 |
| Resume queries | 100 |
| LSH candidate pairs | 746 |
| Resumes with at least one match | 89 |
| Top-K output rows | 534 |
| Silver processing | 45.3 seconds |
| Feature processing | 8.5 seconds |
| Retrieval and evaluation | 17.2 seconds |
| Total runtime | 76.1 seconds |

The current threshold gives 89% query coverage. LSH parameters require tuning
before scaling to the entire catalog.

### Proxy evaluation

TF-IDF similarity between each queried resume and its paired labeled job
description produced:

| Metric | Result |
|---|---:|
| ROC-AUC | 0.448 |
| PR-AUC | 0.470 |

This is approximately random discrimination. It means lexical similarity should
not be presented as a predictor of `select` versus `reject`. It does not by
itself measure the relevance of the LinkedIn top-K recommendations because the
dataset contains no relevance labels connecting resumes to those postings.

## Repository structure

```text
Project_JobApps/
|-- AGENTS.md
|-- README.md
|-- environment.yml
|-- requirements.txt
|-- config/
|   |-- local.yaml
|   `-- data_quality.yaml
|-- notebooks/
|   |-- ETL.ipynb
|   `-- setup_pyspark.ipynb
|-- src/jobapps/
|   |-- config.py
|   |-- schemas.py
|   |-- ingestion.py
|   |-- validation.py
|   |-- documents.py
|   |-- features.py
|   |-- retrieval.py
|   |-- evaluation.py
|   |-- transforms/
|   `-- pipelines/
`-- tests/
```

## Environment setup

Anaconda is the primary environment manager:

```powershell
conda env create -f environment.yml
conda activate ds5110
```

The declared environment uses Python 3.11, Java 17, and PySpark 3.5.6.

On the current Windows setup, Spark also requires the local Hadoop binaries:

```powershell
$env:HADOOP_HOME = "C:\hadoop"
$env:hadoop_home_dir = "C:\hadoop"
```

## Run the MVP

The current configuration uses a deterministic 1% job sample and limits query
evaluation to 100 resumes.

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) "src")
python -m jobapps.pipelines.mvp_pipeline `
  --config config/local.yaml `
  --quality-config config/data_quality.yaml
```

Run the test suite with:

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) "src")
python -m pytest tests -q
```

Current verification result: 16 tests passed.

## Limitations

- TF-IDF is lexical and does not deeply represent semantic equivalence.
- The LSH distance threshold currently leaves some resume queries unmatched.
- The labeled resumes are not directly linked to the LinkedIn postings.
- Resume PII masking uses deterministic regular expressions and synthetic-data
  header rules; it is not a production PII detection system.
- Not every rule in `data_quality.yaml` is executed as a formal validation
  report yet.
- Full-catalog execution and multi-scale benchmarking remain to be performed.
- `notebooks/setup_pyspark.ipynb` is a legacy S3 ingestion notebook. Its Hadoop
  and AWS configuration paths are sanitized placeholders that must be updated
  before another user runs it; it does not contain credential values.

## Next steps

1. Review recommendations manually and create a small relevance judgment set.
2. Tune LSH bucket length, hash-table count, and distance threshold for recall,
   candidate volume, and latency.
3. Benchmark the same pipeline at 0.1%, 1%, 10%, and full-catalog scales.
4. Add explicit title similarity and resume/job skill-overlap features.
5. Compare the TF-IDF baseline with pretrained transformer embeddings for
   semantic retrieval.
6. Use a cross-encoder or supervised reranker only after obtaining defensible
   resume-job relevance labels.
7. Execute all configured data-quality rules and persist a validation report.

The strongest immediate modeling step is pretrained transformer embeddings,
compared against this TF-IDF baseline using the same queries and relevance set.
