# Zepto Data, Analytics, and Support Assistant Project

## Overview

This repository contains three connected data/AI engineering modules. Each module has its own folder, source files, outputs, and README evidence. The project is designed to run without any paid API keys.

| Module | Folder | Main outcome |
|---|---|---|
| 1. Data Pipeline | `data_pipeline/` | Scraped, cleaned, converted, and normalized product data in SQLite |
| 2. Analytics Pipeline | `analytics/` | Titanic EDA, classification, tuning, regression, and saved model pipeline |
| 3. Support Assistant | `support_assistant/` | ChromaDB + LangGraph + FastAPI policy-question RAG service |

## Installation

Use Python 3.10 or later. Install dependencies once from the repository root:

```bash
python -m pip install -r requirements.txt
```

For Google Colab, upload the corresponding `.ipynb` file and run all cells. Colab may show dependency or Hugging Face unauthenticated-download warnings; these are expected when the cell completes without a traceback.

## Module 1 — Data Pipeline

### Objective

This module demonstrates a complete raw-to-relational pipeline using the public Books to Scrape practice site. It scrapes three book categories—Travel, Mystery, and Historical Fiction—so the final dataset contains at least 60 books.

### Pipeline steps

1. `pipeline.py` requests and parses the catalogue pages with `requests` and `BeautifulSoup`.
2. It captures `title`, listed `price`, text `star_rating`, listed `availability`, and `category` for each book.
3. It cleans data safely: price becomes float `price_gbp`; One–Five ratings become integer `rating`; availability becomes boolean `in_stock`.
4. Malformed numeric values are median-imputed to prevent a messy row from breaking the pipeline. Rows with no title/category are dropped because those essential relational fields cannot be inferred responsibly.
5. `price_inr` is calculated using the assignment-defined, fixed baseline **1 GBP = 105.50 INR**. This is not a live market rate.
6. The script creates a SQLite database with two normalized tables:

```text
categories(category_id PK, category_name UNIQUE)
        1 ──────── many
books(book_id PK, ..., category_id FK → categories.category_id)
```

7. It saves raw/clean CSVs, the SQLite database, and SQL evidence in `data_pipeline/artifacts/`.

### Run

```bash
python data_pipeline/pipeline.py
```

The script is expected to report 69 books across three categories. `query_results.md` records five executed SQL queries covering `SELECT`, `WHERE`, `ORDER BY`, `LIMIT`, `DISTINCT`, `BETWEEN`, `IN`, and `JOIN`. It also demonstrates that a SQL JOIN and an in-memory `pandas.merge` produce equivalent results.

## Module 2 — Analytics Pipeline

### Objective

This module builds one cohesive EDA and machine-learning pipeline from Seaborn's Titanic dataset. It is loaded once through `sns.load_dataset('titanic')`; all later stages use the same cleaned DataFrame and the committed `titanic.csv` fallback.

### Data quality decisions

The notebook prints `info()`, `describe()`, shape, and missing-value percentages before cleaning.

- `deck` has more than 30% missing values, so it is dropped. Imputing mostly absent cabin information would be unreliable.
- `age` lies between 5% and 30% missing, so it is median-imputed.
- `embarked` and its duplicate display-name field `embark_town` have fewer than 5% missing rows. Those rows are dropped.

### EDA and interpretation

The notebook includes age/fare histograms and box plots, IQR outlier counts, and mean/median/mode for fare. Fare is right-skewed because mean > median > mode. It reports survival rates by sex, passenger class, and their combination using groupby and boolean masking.

The correlation heatmap uses exactly: `survived`, `pclass`, `age`, `sibsp`, `parch`, and `fare`. `adult_male` and `alone` are excluded because they are derived/redundant variables. Four multivariate charts describe how sex, class, fare, and age relate to survival. The notebook also performs an EDA-only z-score check, showing transformed age/fare means near 0 and standard deviations near 1; this transformation is not reused for modelling.

### Modelling

The target is `survived`. A stratified train/test split happens before all preprocessing, preserving the survival-class balance. A scikit-learn `ColumnTransformer` inside each pipeline performs median imputation, categorical imputation, one-hot encoding, and numeric scaling. Therefore, every preprocessing step fits on the training split only and only transforms the test split.

Three classifiers are trained on the same split:

- Logistic Regression
- Decision Tree (with a labelled `plot_tree` chart)
- Random Forest

For each classifier, the executed notebook displays confusion matrix, accuracy, precision, recall, F1, ROC curve, and AUC in a comparison table. It also compares baseline Logistic Regression, `class_weight='balanced'`, and training-fold-only SMOTE using precision/recall/F1. Random Forest tuning uses `GridSearchCV` for `n_estimators`, `max_depth`, and `max_features`, and reports the valid out-of-bag score from a model created with `oob_score=True`.

The regression side-task predicts `fare` using other non-leaking features. It reports MAE, RMSE, R², Adjusted R², and a residual plot. Classification and regression metrics are deliberately displayed in separate tables because they are not comparable scales.

The final cell saves and reloads `best_titanic_pipeline.joblib`. This artifact includes preprocessing plus the final classifier, so it can predict from raw new feature rows.

### Colab run

Upload `analytics_titanic_colab.ipynb` to Google Colab and choose **Runtime → Run all**. Download the generated `titanic.csv` and `best_titanic_pipeline.joblib` after completion.

## Module 3 — Support Assistant

### Objective

This module is a small RAG service that answers only from the supplied Zepto delivery, returns, membership, tracking, cancellation, gift-card, and support policies.

### Architecture

```text
docs/doc_01.txt ... docs/doc_08.txt
        ↓ ingestion and one-document chunks
all-MiniLM-L6-v2 local embeddings
        ↓
ChromaDB collection: zepto_policy_chunks
        ↓
LangGraph: classify_intent
   ├─ policy_question  → retrieve_and_answer → validated JSON
   └─ general_question → direct_answer       → validated JSON
```

`collection()` reads all eight local files, embeds each short document with the open-source `sentence-transformers` model `all-MiniLM-L6-v2`, and saves vectors/metadata in ChromaDB. For policy questions, `retrieve_and_answer` embeds the query, retrieves three cosine-similar chunks, and uses the highest-ranked retrieved chunk in the answer.

The graph has the three named nodes required by the assignment: `classify_intent`, `retrieve_and_answer`, and `direct_answer`. `classify_intent` routes policy terms such as delivery, return, refund, membership, tracking, cancel, gift card, and support hours to retrieval; unrelated questions go directly to the policy-only response.

### Mock mode and structured output

`MOCK_LLM` defaults to `1`, the required graded baseline. It is fully deterministic, requires no account/API key, and makes no provider call. In mock mode, a policy response starts with `Based on the retrieved context:` and includes the retrieved document IDs. A general query returns the fixed message: `I can only answer questions about Zepto policies right now.`

Every response is validated with Pydantic:

```json
{
  "answer": "string",
  "sources": ["doc_01"],
  "confidence": 1.0
}
```

`PROMPT_TEMPLATE` provides the optional `MOCK_LLM=0` extension with all required prompt components: role, context, task, output format, response length, explicit negative constraint, and a few-shot example. The default mock path is the path intended for assessment.

### Run locally and test

```bash
cd support_assistant
uvicorn main:app --host 0.0.0.0 --port 7860
```

Then call the endpoint:

```bash
curl -X POST http://127.0.0.1:7860/ask -H "Content-Type: application/json" -d '{"query":"What is the delivery fee?"}'
curl -X POST http://127.0.0.1:7860/ask -H "Content-Type: application/json" -d '{"query":"What is the capital of India?"}'
```

The first call exercises retrieval; the second exercises the general-question branch. HTTP status `200` for both confirms the FastAPI route is working.

### Docker

```bash
cd support_assistant
docker build -t zepto-support-assistant .
docker run --rm -p 7860:7860 zepto-support-assistant
```

The provided Dockerfile runs `uvicorn main:app --host 0.0.0.0 --port 7860`, exposing `POST /ask` locally. The first embedding run downloads the open-source model from Hugging Face; the unauthenticated-download warning is expected and does not require an HF token.
