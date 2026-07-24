# Job Application Retrieval Pipeline

This project builds a distributed ETL and NLP pipeline for matching resumes to
job postings. The current MVP is an unsupervised information-retrieval system:
given a resume, it retrieves and ranks similar jobs from a LinkedIn job catalog.

The system is decision support and must not be used to make autonomous hiring
decisions.

## Quick start

The commands below use Anaconda and Windows PowerShell. The public source-data
download is approximately 3.1 GB and does not require an AWS account.

```powershell
git clone https://github.com/Ratocracy/jobapp_project.git
cd jobapp_project

conda env create -f environment.yml
conda activate ds5110

$env:PYTHONPATH = (Join-Path (Get-Location) "src")
python -m jobapps.download_data
```

Verify the installation:

```powershell
python -m pytest tests -q
```

Run the configured 1% jobs/100-resume MVP:

```powershell
# Update this path if Hadoop is installed elsewhere.
$env:HADOOP_HOME = "C:\hadoop"
$env:hadoop_home_dir = "C:\hadoop"

python -m jobapps.pipelines.mvp_pipeline `
  --config config/local.yaml `
  --quality-config config/data_quality.yaml
```

Generated silver/gold tables are written under `data/`. Source and generated
datasets are intentionally excluded from Git.

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

First, download and checksum-verify the four source Parquets from the public S3
bucket. No AWS account or credentials are required:

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) "src")
python -m jobapps.download_data
```

Files are downloaded to `raw_data/` through temporary `.part` files. Existing
valid files are skipped. Existing files that fail verification are preserved
unless `--force` is explicitly supplied.

To verify an existing local copy without contacting S3:

```powershell
python -m jobapps.download_data --verify-only
```

Individual datasets can be requested by repeating `--dataset`:

```powershell
python -m jobapps.download_data `
  --dataset linkedin_job_postings `
  --dataset job_summary
```

The expected object names, byte sizes, and SHA-256 checksums are versioned in
`config/data_manifest.yaml`. A private mirror can still be accessed with
`--profile PROFILE_NAME` or `--use-default-chain`.

After the source files are present, run the MVP:

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

Current verification result: 28 tests passed.

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
2. Validate the selected retrieval settings at 5%, 10%, and full-catalog scales.
3. Benchmark exact and approximate retrieval latency at each catalog scale.
4. Add explicit title similarity and resume/job skill-overlap features.
5. Compare the TF-IDF baseline with pretrained transformer embeddings for
   semantic retrieval.
6. Use a cross-encoder or supervised reranker only after obtaining defensible
   resume-job relevance labels.
7. Execute all configured data-quality rules and persist a validation report.

The strongest immediate modeling step is pretrained transformer embeddings,
compared against this TF-IDF baseline using the same queries and relevance set.

## Retrieval tuning

We measured LSH recall against exhaustive cosine retrieval before changing the
text representation. The controlled benchmark uses a deterministic 1% catalog
(13,634 jobs), the same 100 resume queries, normalized TF-IDF vectors, and
top-10 retrieval. Exact sparse matrix multiplication evaluates all 1,363,400
resume-job pairs without converting the 262,144-position vectors to dense Spark
arrays.

The original LSH settings recovered only 51.8% of the exact top-10 results:

| Bucket length | Hash tables | Distance threshold | Candidates | Coverage | Recall@10 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.8 | 3 | 1.2 | 746 | 89/100 | 51.8% |

We then evaluated 60 combinations:

```text
bucket_length      = [0.5, 0.8, 1.0, 1.2, 1.5]
num_hash_tables    = [3, 5, 8, 12]
distance_threshold = [1.2, 1.3, 1.4]
```

For each bucket/hash model, candidates were generated once at threshold 1.4;
the 1.2 and 1.3 results were evaluated as nested distance subsets. Reported
per-configuration runtime therefore includes model fitting and generation at
1.4 and should not be interpreted as standalone latency for the lower
thresholds.

Key results:

| Operating point | Bucket | Hashes | Threshold | Candidates | Coverage | Recall@10 | Runtime |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Original | 0.8 | 3 | 1.2 | 746 | 89/100 | 51.8% | 79.6 s |
| Practical | 0.5 | 3 | 1.3 | 4,885 | 97/100 | 92.4% | 70.2 s |
| Strict target | 0.5 | 3 | 1.4 | 349,324 | 100/100 | 95.4% | 70.6 s |
| Maximum recall | 0.8 | 8 | 1.4 | 390,874 | 100/100 | 100.0% | 103.5 s |

```text
Recall
100 |                           ●
 95 |                      ●
 90 |                 ●
 85 |
 80 |
 70 |
 60 |      ●
 50 | ●
    +-----------------------------
      700   5k    50k    400k
          Candidate pairs
```

At threshold 1.3, increasing from 3 to 8 hash tables improved recall from
92.4% to 96.6% but did not improve 97/100 query coverage. The remaining three
queries had no candidates within the distance threshold, so additional hash
tables could not recover them. Raising the threshold to 1.4 achieved complete
coverage but increased the candidate set from thousands to hundreds of
thousands.

Bucket length produced no measurable candidate-count or recall difference in
this grid. Runtime differences across bucket lengths are therefore treated as
execution noise rather than evidence for one value.

Exact sparse cosine took 6.1 seconds at 1% and was faster than every LSH grid
configuration at this scale. The practical next design is threshold 1.3 with a
fallback for uncovered queries, followed by validation at larger catalog sizes
where approximate retrieval may provide a runtime benefit.

Run the exact baseline and tuning grid with:

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) "src")
python -m jobapps.pipelines.retrieval_benchmark
python -m jobapps.pipelines.retrieval_tuning
```

Machine-readable results are written to ignored paths under
`data/gold/benchmarks/`.

## Transformer benchmark

The transformer implementation is a parallel gold-feature experiment; it does
not replace the silver ETL or the TF-IDF baseline. It uses the same
deterministic 1% job catalog and 100 resume queries so the two representations
can be compared without LSH approximation.

```text
shared combined_text
        |
        +--> TF-IDF -------------> exact sparse cosine top-10
        |
        `--> overlapping chunks
                  |
                  v
          all-MiniLM-L6-v2
                  |
                  v
          mean-pooled, normalized
          384-dimensional embedding
                  |
                  v
          exact dense cosine top-10
```

Long documents are split using the model tokenizer into overlapping 240-token
chunks with a 32-token overlap. Each chunk is encoded and normalized. Chunk
vectors are mean-pooled and normalized again to create one vector per job or
resume. This avoids silently using only the beginning of long descriptions.

Generated embeddings are cached under:

- `data/gold/benchmarks/transformer_1pct/job_embeddings`
- `data/gold/benchmarks/transformer_1pct/resume_embeddings`

The benchmark also writes transformer and TF-IDF top-10 tables plus
`metrics.json`. These generated files and downloaded model weights are not
tracked by Git.

Install or update the declared environment before the first run:

```powershell
conda env update -f environment.yml --prune
conda activate ds5110
$env:PYTHONPATH = (Join-Path (Get-Location) "src")
```

Run the controlled comparison:

```powershell
python -m jobapps.pipelines.transformer_benchmark `
  --config config/transformer_1pct.yaml `
  --quality-config config/data_quality.yaml
```

The first execution downloads
`sentence-transformers/all-MiniLM-L6-v2` and creates the cached embeddings.
Later executions reuse a cache only when its row counts and model name match
the configuration. Use `--force-recompute` after changing chunking logic or
when a fresh embedding run is required.

Top-K overlap measures how differently TF-IDF and the transformer retrieve; it
does not determine which system is better. Choosing between them still requires
manual relevance judgments or a defensible labeled resume-job dataset.
