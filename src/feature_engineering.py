from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.sql import DataFrame
from pyspark.sql.functions import col


def add_grade_index(df: DataFrame) -> DataFrame:
    indexer = StringIndexer(inputCol="grade", outputCol="label", handleInvalid="skip")
    return indexer.fit(df).transform(df)


def add_class_weights(df: DataFrame) -> DataFrame:
    """Create inverse-frequency weights matching the original RF workflow."""
    total = df.count()
    n_classes = df.select("label").distinct().count()
    counts = df.groupBy("label").count().collect()
    weights = {float(r["label"]): total / (n_classes * r["count"]) for r in counts}

    # Keep this implementation dependency-light and equivalent to the original UDF mapping.
    from pyspark.sql.functions import create_map, lit
    mapping = create_map(*[x for pair in weights.items() for x in (lit(pair[0]), lit(float(pair[1])))])
    return df.withColumn("classWeightCol", mapping[col("label")])


def encode_features(df: DataFrame):
    categorical_cols = [
        name for name, dtype in df.dtypes
        if dtype == "string" and name != "grade"
    ]
    indexers = [
        StringIndexer(inputCol=c, outputCol=f"{c}_indexed", handleInvalid="keep")
        for c in categorical_cols
    ]
    indexed = Pipeline(stages=indexers).fit(df).transform(df)

    encoder = OneHotEncoder(
        inputCols=[f"{c}_indexed" for c in categorical_cols],
        outputCols=[f"{c}_ohe" for c in categorical_cols],
    )
    numeric_cols = [
        name for name, dtype in indexed.dtypes
        if dtype in {"double", "int"} and name not in {"label", "classWeightCol"}
    ]
    assembler = VectorAssembler(
        inputCols=numeric_cols + [f"{c}_ohe" for c in categorical_cols],
        outputCol="features",
    )
    result = Pipeline(stages=[encoder, assembler]).fit(indexed).transform(indexed)
    return result, numeric_cols, categorical_cols, assembler
