# Text Classifier Embedding Extraction

Fine-tunes a transformer classifier end-to-end on 9 standard text classification
benchmarks and dumps per-example features, logits, softmax predictions, and labels
for each split — intended as input for downstream quantification experiments
(e.g. with [QuaPy](https://github.com/HLT-ISTI/QuaPy)), but usable standalone.

## Datasets

| name | source | classes |
|---|---|---|
| imdb | stanfordnlp/imdb | 2 |
| rt | cornell-movie-review-data/rotten_tomatoes | 2 |
| yelp | Yelp/yelp_review_full | 5 |
| ag_news | fancyzhx/ag_news | 4 |
| sst5 | SetFit/sst5 | 5 |
| 20newsgroups | SetFit/20_newsgroups | 20 |
| dbpedia | fancyzhx/dbpedia_14 | 14 |
| yahoo_answers | community-datasets/yahoo_answers_topics | 10 |
| trec | SetFit/TREC-QC (coarse labels) | 6 |

All datasets are loaded via Hugging Face `datasets` and kept in their original,
non-binarized form.

## Setup and run

```bash
bash setup_and_run.sh [conda_env_name]   # default env name: text_embed

## Reshape Data

The script reshape_data.py unpacks the npz files and rewrites then split by modality so it is more efficient for data loading.
The reesulting data are uploaded to zenodo.

