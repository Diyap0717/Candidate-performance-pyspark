from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics


def evaluate(predictions):
    evaluator = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="accuracy"
    )
    f1 = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="f1"
    )
    return evaluator.evaluate(predictions), f1.evaluate(predictions)


def confusion_matrix(predictions):
    pairs = predictions.select("prediction", "label").rdd.map(tuple)
    return MulticlassMetrics(pairs).confusionMatrix().toArray()
