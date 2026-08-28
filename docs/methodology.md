# Methodology Notes

## Original workflow

The original notebook loaded the CSV with Spark, corrected `math_score`, removed `roll_no`, removed null rows, checked duplicates, explored the target distribution, indexed the grade label, calculated inverse-frequency class weights, encoded categorical variables, assembled features, split the data 80/20, and trained Spark ML classifiers.

The original report states that the cleaned dataset contained 9,787 records and that the split contained 7,906 training records and 1,881 test records.

## Model comparison

The course notebook compared Random Forest, Logistic Regression, and Decision Tree. The reported test results were:

| Model | Accuracy | F1 |
|---|---:|---:|
| Random Forest | 0.9096 | 0.8955 |
| Logistic Regression | 0.9984 | 0.9984 |
| Decision Tree | 0.9724 | 0.9714 |

The notebook's initial Random Forest workflow explicitly used `classWeightCol`. The later model-comparison loop did not pass that weight column to the Logistic Regression or Decision Tree estimators. Therefore, the repository does not describe all three comparison models as being class-weighted.

## Leakage warning

The feature set includes `total_score`, while the report describes grade thresholds based on total score. The notebook's Random Forest feature analysis also shows `total_score` as the dominant feature. This is a strong signal that the task, as originally formulated, contains target leakage.

For a portfolio-quality follow-up experiment, the correct question is not merely "Can we predict grade?" but "Can we predict grade without using a feature that directly determines the grade?" That experiment requires the original dataset and is therefore not fabricated in this repository.
