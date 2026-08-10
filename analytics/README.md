# Module 2 — Titanic Analytics Pipeline

## Run order and offline fallback

Run `analytics_titanic_colab.ipynb` from top to bottom in Google Colab. It calls `sns.load_dataset('titanic')` exactly once, then saves the cleaned shared dataset as `titanic.csv`. Every EDA, classifier, tuning, regression, and persistence step continues from that same DataFrame/CSV; there is no second dataset load. The notebook also saves `best_titanic_pipeline.joblib`, the complete fitted preprocessing-plus-model pipeline.

## Data quality decisions

The initial profile reports the shape, `info()`, `describe()`, and missing-value percentages. The original Titanic data has approximately 77.22% missing `deck`, 19.87% missing `age`, and 0.22% missing each of `embarked` and its display-name duplicate `embark_town`.

- `deck` exceeds the 30% threshold, so it is dropped: cabin-deck imputation would mostly invent data.
- `age` lies in the 5–30% band, so it is median-imputed.
- `embarked`/`embark_town` lie below 5%, so the two affected passenger rows are dropped together.

This follows the required threshold rule and leaves a clean dataset of 889 rows. Fare is right-skewed: its mean is greater than its median, which is greater than its mode. The notebook reports IQR outlier counts for both `age` and `fare`, and its histograms/box plots make the fare tail visible.

## EDA story and interpretations

1. **Sex and passenger class bar chart.** Female passengers survived at much higher rates than male passengers in every class. First class also has a survival advantage, so sex and class together give a stronger explanation than either variable alone.
2. **Fare/class/survival box plot.** Fares differ substantially by passenger class; first-class passengers generally paid more. Within each class, survival groups overlap, so fare is informative but is not a complete explanation of survival.
3. **Age–fare scatter plot, colored by survival and styled by sex.** Survivors appear at many ages, showing that age alone does not separate the target. The plot makes the stronger sex pattern visible while retaining the class/fare-related spread.
4. **Six-variable correlation heatmap.** The heatmap uses exactly `survived`, `pclass`, `age`, `sibsp`, `parch`, and `fare`. The notebook ranks every off-diagonal pair by absolute correlation and prints the top two, which must be cited from the executed output. `adult_male` and `alone` are excluded because they are derived/redundant fields rather than independent measured variables.

The notebook also performs an EDA-only z-score check on `age` and `fare`. Their transformed means are approximately zero and population standard deviations approximately one. This check does not feed the model; model scaling occurs separately inside training-only pipelines.

## Modelling and evaluation

`survived` is the target. The data is split before preprocessing using `train_test_split(..., stratify=y)`, so train and test keep similar survived/not-survived proportions. Median imputation, most-frequent categorical imputation, one-hot encoding, and numeric scaling are inside a `ColumnTransformer`/`Pipeline`; they are fit on training data only and transform the test data without refitting.

The notebook trains Logistic Regression, Decision Tree, and Random Forest on the identical split. It provides a confusion matrix, accuracy, precision, recall, F1, ROC curve, and AUC for each, plus a labelled `plot_tree` visualisation for the Decision Tree. The executed notebook prints the required comparison table below; retain those generated values in the submitted notebook.

| Classifier | Accuracy | Precision | Recall | F1 | AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | See executed notebook | See executed notebook | See executed notebook | See executed notebook | See executed notebook |
| Decision Tree | See executed notebook | See executed notebook | See executed notebook | See executed notebook | See executed notebook |
| Random Forest | See executed notebook | See executed notebook | See executed notebook | See executed notebook | See executed notebook |

For imbalance handling, the notebook compares baseline Logistic Regression, `class_weight='balanced'`, and SMOTE. SMOTE is inside an `imblearn` pipeline and is therefore applied only to the training fold. The printed precision/recall/F1 table identifies the winning strategy on held-out F1; this is the appropriate result to cite rather than assuming resampling always improves performance.

Random Forest tuning uses `GridSearchCV` across `n_estimators`, `max_depth`, and `max_features`. The estimator is constructed with `oob_score=True`, so the notebook reports both the selected parameters and the valid OOB score.

## Regression side-task

A separate multivariate Linear Regression predicts `fare` from the other non-leaking available features. Its MAE, RMSE, R², and Adjusted R² appear in a distinct table because they are not on the same scale as classification scores. The residual plot is the heteroscedasticity diagnostic: a fan-shaped or systematically changing residual spread is evidence of heteroscedasticity; an even, random band around zero is not.

## Final recommendation

Deploy the classifier with the highest held-out **F1** shown by the executed comparison table, while also checking its AUC, precision, and recall. F1 is a useful balance of precision and recall for this moderately imbalanced survival target; AUC adds threshold-independent separation evidence. The saved `best_titanic_pipeline.joblib` is the entire fitted pipeline, not a bare estimator, and the final notebook cell reloads it and predicts directly on raw, unprocessed feature rows.

