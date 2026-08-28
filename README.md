# Scalable Student Grade Classification with PySpark

A distributed machine-learning workflow for classifying student grades using **Apache Spark (PySpark)**. The project combines data cleaning, exploratory analysis, class-imbalance handling, categorical feature encoding, multiclass model benchmarking, feature analysis, and a proposed AWS deployment architecture.

> **Important methodological note:** The original course analysis achieved very high test performance, but `total_score` is directly tied to the subject scores and the grade thresholds. That makes it a target-leakage variable for a genuine early-prediction use case. Rather than hiding this limitation, this repository documents it as a central finding and separates the reported course results from what would be required for a leakage-free production model.

## Why this project

The course project was designed to explore how distributed data processing and Spark ML can be combined into a repeatable classification workflow for student academic outcomes. The work uses a 10,000-row student-performance dataset and predicts five grade categories: **A, B, C, D, and Fail**.

The portfolio version focuses on the engineering and analytical lessons from the work rather than presenting the original accuracy as production-ready evidence.

## What was done

- Loaded and processed the dataset with PySpark.
- Corrected the `math_score` data type and removed the identifier column.
- Audited missing values and removed rows containing nulls, leaving 9,787 records.
- Checked for duplicate records; none were found.
- Explored score distributions and grade imbalance.
- Indexed and one-hot encoded categorical variables.
- Built Spark ML feature vectors with `VectorAssembler`.
- Addressed class imbalance with inverse-frequency class weights in the Random Forest workflow.
- Compared Random Forest, Logistic Regression, and Decision Tree classifiers.
- Evaluated models using accuracy, F1 score, and confusion matrices.
- Performed Random Forest hyperparameter search with Spark `CrossValidator`.
- Examined Random Forest feature importance.
- Documented a theoretical S3 → Airflow → EMR/Spark → S3 → Athena → QuickSight architecture.

## Dataset

The source report describes the dataset as 10,000 anonymized high-school-student records containing demographic, socioeconomic, preparation, subject-score, total-score, and grade fields. The report states that the data are synthetic and inspired by real-world trends.

The original dataset is **not included in this repository** because it was not provided with the project files used to prepare this portfolio version. See [`data/README.md`](data/README.md).

## Results from the original course analysis

| Model | Accuracy | F1 Score |
|---|---:|---:|
| Random Forest | 0.9096 | 0.8955 |
| Logistic Regression | 0.9984 | 0.9984 |
| Decision Tree | 0.9724 | 0.9714 |

These numbers should be interpreted carefully. The original feature set includes `total_score`, while `grade` is determined from total-score ranges. The resulting performance therefore should **not** be interpreted as evidence that the model can reliably predict a student's future grade before those component scores are available.

## Key findings

### 1. Severe class imbalance

The cleaned dataset contained:

- A: 895
- B: 5,544
- C: 2,628
- D: 659
- Fail: 61

The `Fail` class represented only about 0.6% of the cleaned records, motivating class weighting in the Random Forest workflow.

### 2. Academic scores dominate the prediction

The original Random Forest analysis identified `total_score` as the dominant feature, followed by science and math scores. Demographic variables had substantially smaller importance scores.

### 3. Test preparation showed only a small difference

The exploratory analysis found only a marginal difference in average score between students who completed the preparation course and those who did not.

### 4. The most important lesson: target leakage

Because grade is assigned from total-score thresholds and `total_score` is derived from the subject scores, including `total_score` makes the prediction task partly circular. A stronger next experiment would remove `total_score` and evaluate whether demographic and preparation variables provide useful predictive signal independently of the final aggregate score.

## Architecture

The implemented workflow is a local/Colab Spark ML workflow. The report also proposes a theoretical cloud architecture:

```text
Raw Data
   ↓
Amazon S3
   ↓
Apache Airflow
   ↓
Amazon EMR / Spark
   ↓
Predictions → Amazon S3
   ↓
Amazon Athena
   ↓
Amazon QuickSight
```

This AWS design is **conceptual only**; it was not deployed as part of the course project.

## Repository structure

```text
.
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   ├── student_performance_analysis.ipynb
│   └── original_course_notebook.ipynb
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── modeling.py
│   └── evaluation.py
├── sql/
│   └── athena_schema.sql
├── data/
│   └── README.md
└── docs/
    └── methodology.md
```

## Running the project

The notebook was originally developed in Google Colab with a locally mounted Google Drive dataset. The portfolio version removes that dependency and expects the dataset path to be supplied through `DATA_PATH`.

```bash
pip install -r requirements.txt
```

Then set the dataset path and run the notebook in a Spark-enabled environment.

```bash
# Linux/macOS
export DATA_PATH="/path/to/Student_performance_10k.csv"

# Windows PowerShell
$env:DATA_PATH="C:\path\to\Student_performance_10k.csv"
```

The repository does not claim a fully reproducible run because the original dataset was not supplied with the two source files used for this cleanup.

## Limitations and next steps

1. Remove `total_score` from the modeling feature set and rebuild the evaluation.
2. Compare per-class precision, recall, and F1, especially for the rare `Fail` class.
3. Use stratified or otherwise carefully designed validation where appropriate.
4. Tune models inside a complete Spark ML pipeline rather than tuning a detached estimator.
5. Investigate whether the dataset's synthetic construction limits generalization to real educational data.
6. If deployed, add monitoring, data-quality checks, model-versioning, and an inference API.

## Skills demonstrated

**Data Engineering:** PySpark, distributed transformations, Spark ML pipelines, data-quality checks, scalable architecture design.

**Machine Learning:** multiclass classification, categorical encoding, class weighting, model comparison, cross-validation, evaluation.

**Data Science:** exploratory analysis, feature relationships, imbalance analysis, model interpretation, and identification of target leakage.

## Course context

Originally completed as a final project for **CSC 555 – Mining Big Data** during an MS Data Science program.
