from pyspark.ml.classification import (
    DecisionTreeClassifier,
    LogisticRegression,
    RandomForestClassifier,
)
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder


def train_random_forest(train_df):
    model = RandomForestClassifier(
        featuresCol="features",
        labelCol="label",
        weightCol="classWeightCol",
        numTrees=100,
        maxDepth=5,
        seed=42,
    )
    return model.fit(train_df)


def compare_models(train_df, test_df):
    models = {
        "RandomForest": RandomForestClassifier(labelCol="label", featuresCol="features"),
        "LogisticRegression": LogisticRegression(labelCol="label", featuresCol="features", maxIter=50),
        "DecisionTree": DecisionTreeClassifier(labelCol="label", featuresCol="features"),
    }
    accuracy = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="accuracy"
    )
    f1 = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="f1"
    )
    results, fitted = [], {}
    for name, estimator in models.items():
        model = estimator.fit(train_df)
        pred = model.transform(test_df)
        results.append((name, accuracy.evaluate(pred), f1.evaluate(pred)))
        fitted[name] = model
    return results, fitted


def tune_random_forest(train_df):
    rf = RandomForestClassifier(labelCol="label", featuresCol="features")
    grid = (
        ParamGridBuilder()
        .addGrid(rf.maxDepth, [3, 5, 10])
        .addGrid(rf.numTrees, [10, 20, 30])
        .addGrid(rf.maxBins, [16, 32])
        .build()
    )
    evaluator = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="accuracy"
    )
    cv = CrossValidator(
        estimator=rf,
        estimatorParamMaps=grid,
        evaluator=evaluator,
        numFolds=2,
    )
    return cv.fit(train_df)
