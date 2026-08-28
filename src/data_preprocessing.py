from pyspark.sql import DataFrame
from pyspark.sql.functions import col, sum as spark_sum


def load_data(spark, path: str) -> DataFrame:
    return spark.read.csv(path, header=True, inferSchema=True)


def clean_data(df: DataFrame) -> DataFrame:
    """Apply the cleaning steps used in the original course analysis."""
    df = df.withColumn("math_score", col("math_score").cast("integer"))
    df = df.drop("roll_no")
    df = df.na.drop()
    return df.dropDuplicates()


def null_counts(df: DataFrame) -> DataFrame:
    return df.select(
        [spark_sum(col(c).isNull().cast("int")).alias(c) for c in df.columns]
    )
