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

Run the configured sampled MVP:

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

The production MVP remains the classical NLP baseline:

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

A parallel development benchmark now adds distributed transformer embeddings
and unigram-plus-bigram TF-IDF. There is still no cross-encoder or supervised
learning-to-rank model.

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

Benchmark configurations use a deterministic 1% job sample and 100 validation
resume queries. `config/local.yaml` is the larger 5%/200-query scaling
configuration.

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

Current verification result: 36 tests passed.

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
2. Complete blinded relevance judgments on validation recommendations.
3. Freeze model and retrieval settings before generating locked-test queries.
4. Validate the selected retrieval settings at 5%, 10%, and full-catalog scales.
5. Add explicit title similarity and resume/job skill-overlap features.
6. Use a cross-encoder or supervised reranker only after obtaining defensible
   resume-job relevance labels.
7. Execute all configured data-quality rules and persist a validation report.

The immediate modeling step is a three-way validation comparison among unigram
TF-IDF, unigram-plus-bigram TF-IDF, and transformer embeddings.

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

The transformer implementation is a gold-feature experiment; it does not
replace the silver ETL or the TF-IDF baseline. It uses the same
deterministic 1% job catalog and 100 resume queries so the two representations
can be compared without LSH approximation.
The queries come only from the validation split.

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
Spark prepares the typed, leakage-safe document tables. For the controlled 1%
sample, the driver collects only the selected document contract and
SentenceTransformers performs batched PyTorch inference with one MiniLM model.
The resulting embeddings are immediately converted back into a Spark DataFrame
for Parquet persistence, retrieval, joins, and evaluation.

This hybrid boundary is deliberate. Spark remains responsible for ingestion,
cleaning, sampling, splits, lexical features, and downstream evaluation, while
PyTorch handles the neural-network operation for which it is optimized.

Generated embeddings are cached under:

- `data/gold/benchmarks/transformer_1pct_validation/job_embeddings`
- `data/gold/benchmarks/transformer_1pct_validation/resume_embeddings`

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
Later executions reuse a cache only when its model name and complete source-ID
set match the configured documents. Use `--force-recompute` after changing
chunking logic or when a fresh embedding run is required.

Top-K overlap measures how differently TF-IDF and the transformer retrieve; it
does not determine which system is better. Choosing between them still requires
manual relevance judgments or a defensible labeled resume-job dataset.

The split-correct three-way validation comparison uses the same 100 resume IDs
for every representation. Pairwise top-10 overlaps are:

| Representation pair | Mean top-10 overlap |
| --- | ---: |
| Unigram TF-IDF vs unigram + bigram TF-IDF | 73.1% |
| Unigram TF-IDF vs MiniLM | 40.4% |
| Unigram + bigram TF-IDF vs MiniLM | 39.7% |

### Why transformer inference runs in PyTorch

A distributed Spark `mapInPandas` implementation was tested locally to evaluate
whether transformer inference should also run inside Spark workers:

| Execution | CPU layout | Peak benchmark RAM | Outcome |
| --- | ---: | ---: | --- |
| Driver PyTorch | approximately 14 CPU cores | not recorded | Completed on validation queries; embedding took about 710 s |
| Spark workers | 2 partitions x 7 task CPUs | about 3.5 GB | Failed after about 10 min with a Python-worker/Arrow EOF |
| Spark workers | 7 partitions x 2 task CPUs | about 7.1 GB | Failed after about 8 min with the same worker/Arrow EOF |

Both Spark trials remained comfortably within memory limits and reached
sustained CPU utilization. Their failure mode was worker/Arrow stability, not
insufficient RAM. The working MVP therefore uses direct PyTorch inference. The
Spark-worker approach may be revisited on a real cluster with smaller streamed
Arrow batches, but it is not the supported local execution path.

## Development splits and catalog samples

Catalog sampling and evaluation splitting answer different questions:

- `sample_fraction` selects a deterministic subset of jobs by `job_link` for
  runtime and scaling experiments.
- `resume_query_split` selects train, validation, or test resumes from
  leakage-safe job-description groups.

Current development rules:

| Data | Purpose |
| --- | --- |
| Job catalog sample | Fit job-corpus IDF statistics and provide retrieval candidates |
| Train resumes | Reserved for future supervised fitting; not required by unsupervised cosine retrieval |
| Validation resumes | Tune LSH, compare feature representations, and conduct development relevance review |
| Test resumes | Locked until the representation, retrieval settings, and evaluation protocol are frozen |

All development benchmark configs set:

```yaml
resume_query_split: validation
```

Output directories include the split name so validation caches cannot be
mistaken for final test evidence.

## Third model: unigram plus bigram TF-IDF

The phrase-aware lexical model is implemented entirely as a Spark ML pipeline:

```text
RegexTokenizer
-> StopWordsRemover
-> NGram(n=2)
-> prefixed unigram/bigram token union
-> HashingTF (524,288 features)
-> IDF fitted on jobs
-> L2 normalization
```

On the deterministic 1% catalog and 100 validation resumes:

| Metric | Unigram TF-IDF | Unigram + bigram TF-IDF |
| --- | ---: | ---: |
| Feature/retrieval audit time | 16.4 s | 32.8 s |
| Top-10 overlap | — | 73.1% |

The n-gram representation changes about 27% of the top-10 recommendations. The
exact retrieval step is a small-sample audit; production-scale candidate
generation remains Spark LSH.

Run the comparison with:

```powershell
python -m jobapps.pipelines.ngram_benchmark
```

## Blinded three-method relevance review

Generate a pooled labeling packet after all three validation recommendation
tables exist:

```powershell
python -m jobapps.pipelines.build_relevance_review `
  --config config/transformer_1pct.yaml `
  --quality-config config/data_quality.yaml `
  --resume-count 100
```

The command pools unigram, unigram-plus-bigram, and transformer top-10 results,
deduplicates each resume-job pair, and deterministically shuffles candidates.
The generated packet is ignored by Git and written under:

```text
data/gold/benchmarks/three_method_1pct_validation/relevance_review/
```

Reviewers should edit only `review_sheet.csv`. Do not open `answer_key.csv`
until labeling is complete because it contains model identities, ranks, scores,
resume IDs, and job links.

## Course notebooks and EDA

Rubric-aligned notebooks live under `notebooks/final_project/`:

1. `01_data_import_and_preprocessing.ipynb`
2. `02_sampling_and_eda.ipynb`
3. `03_model_construction.ipynb`
4. `04_model_evaluation.ipynb`

They import the tested production modules and explain each call in execution
order. Required EDA figures and their Spark generator are under `reports/eda/`:

- `job_type_counts.png`
- `document_length_distribution.png`

Spark computes the source counts and grouped distributions. Matplotlib renders
only the small aggregated outputs.
