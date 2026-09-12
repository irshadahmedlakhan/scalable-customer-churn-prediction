from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, trim, when
from pyspark.sql.types import DoubleType, IntegerType

from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    StringIndexer,
    OneHotEncoder,
    VectorAssembler,
)


# ============================================================
# PROJECT PATHS
# ============================================================

# data_preprocessing.py is inside:
#
# project/
# └── src/
#     └── data_preprocessing.py
#
# Therefore parents[1] gives the project root automatically.

PROJECT_PATH = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_PATH
    / "data"
    / "Telco-Customer-Churn.csv"
)

OUTPUT_PATH = (
    PROJECT_PATH
    / "data"
    / "processed"
    / "churn_data.parquet"
)


# ============================================================
# START MESSAGE
# ============================================================

print("=" * 70)
print("TELCO CUSTOMER CHURN - DATA PREPROCESSING")
print("=" * 70)

print("\nProject Path:")
print(PROJECT_PATH)

print("\nInput Dataset:")
print(INPUT_PATH)

print("\nProcessed Dataset Output:")
print(OUTPUT_PATH)

print("=" * 70)


# ============================================================
# CHECK INPUT DATASET
# ============================================================

if not INPUT_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found:\n{INPUT_PATH}"
    )


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# CREATE SPARK SESSION
# ============================================================

spark = (
    SparkSession.builder
    .appName("Telco_Churn_Preprocessing")
    .master("local[2]")
    .config(
        "spark.driver.memory",
        "4g",
    )
    .config(
        "spark.sql.shuffle.partitions",
        "2",
    )
    .config(
        "spark.default.parallelism",
        "2",
    )
    .config(
        "spark.hadoop.fs.defaultFS",
        "file:///",
    )
    .config(
        "spark.hadoop.fs.file.impl",
        "org.apache.hadoop.fs.RawLocalFileSystem",
    )
    .config(
        "spark.hadoop.mapreduce.fileoutputcommitter.algorithm.version",
        "2",
    )
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")


print("\nSPARK STARTED SUCCESSFULLY")


# ============================================================
# LOAD RAW DATA
# ============================================================

print()
print("=" * 70)
print("LOADING RAW DATA")
print("=" * 70)


df = spark.read.csv(
    str(INPUT_PATH),
    header=True,
    inferSchema=False,
)


original_rows = df.count()


print(
    "\nOriginal Rows:",
    original_rows,
)

print("\nOriginal Columns:")

print(
    df.columns
)


# ============================================================
# VALIDATE EXPECTED COLUMNS
# ============================================================

expected_columns = [
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
    "Churn",
]


missing_columns = [
    column_name
    for column_name in expected_columns
    if column_name not in df.columns
]


if missing_columns:

    spark.stop()

    raise ValueError(
        "Dataset is missing required columns: "
        f"{missing_columns}"
    )


print(
    "\nDataset column validation successful."
)


# ============================================================
# CLEAN STRING VALUES
# ============================================================

print()
print("=" * 70)
print("CLEANING DATA")
print("=" * 70)


for column_name in df.columns:

    df = df.withColumn(
        column_name,
        when(
            trim(
                col(column_name)
            ) == "",
            None,
        ).otherwise(
            trim(
                col(column_name)
            )
        ),
    )


print(
    "Blank string values converted to null."
)


# ============================================================
# CONVERT NUMERIC COLUMNS
# ============================================================

df = df.withColumn(
    "SeniorCitizen",
    col(
        "SeniorCitizen"
    ).cast(
        IntegerType()
    ),
)


df = df.withColumn(
    "tenure",
    col(
        "tenure"
    ).cast(
        IntegerType()
    ),
)


df = df.withColumn(
    "MonthlyCharges",
    col(
        "MonthlyCharges"
    ).cast(
        DoubleType()
    ),
)


df = df.withColumn(
    "TotalCharges",
    col(
        "TotalCharges"
    ).cast(
        DoubleType()
    ),
)


print(
    "Numeric columns converted successfully."
)


# ============================================================
# REMOVE MISSING ROWS
# ============================================================

df = df.dropna()


clean_rows = df.count()


removed_rows = (
    original_rows
    -
    clean_rows
)


print(
    "\nRows after cleaning:",
    clean_rows,
)

print(
    "Rows removed:",
    removed_rows,
)


# ============================================================
# EXPECTED DATASET CHECK
# ============================================================

if original_rows == 7043:

    if clean_rows != 7032:

        print(
            "\nWARNING:"
            " Expected 7,032 clean rows based on the "
            "original Telco dataset, but found:",
            clean_rows,
        )

    else:

        print(
            "\nClean row count verified successfully."
        )


# ============================================================
# FEATURE DEFINITIONS
# ============================================================

categorical_columns = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]


numeric_columns = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]


print()
print("=" * 70)
print("FEATURE DEFINITIONS")
print("=" * 70)

print(
    "\nCategorical Columns:"
)

print(
    categorical_columns
)

print(
    "\nNumeric Columns:"
)

print(
    numeric_columns
)


# ============================================================
# BUILD SPARK ML PIPELINE
# ============================================================

print()
print("=" * 70)
print("BUILDING FEATURE PIPELINE")
print("=" * 70)


stages = []


# ============================================================
# CATEGORICAL STRING INDEXERS
# ============================================================

indexed_columns = []


for column_name in categorical_columns:

    indexed_column = (
        column_name
        +
        "_index"
    )

    indexer = StringIndexer(
        inputCol=column_name,
        outputCol=indexed_column,
        handleInvalid="keep",
    )

    stages.append(
        indexer
    )

    indexed_columns.append(
        indexed_column
    )


# ============================================================
# ONE-HOT ENCODER
# ============================================================

encoded_columns = [
    column_name
    +
    "_encoded"
    for column_name in categorical_columns
]


encoder = OneHotEncoder(
    inputCols=indexed_columns,
    outputCols=encoded_columns,
)


stages.append(
    encoder
)


# ============================================================
# VECTOR ASSEMBLER
# ============================================================

feature_columns = (
    numeric_columns
    +
    encoded_columns
)


assembler = VectorAssembler(
    inputCols=feature_columns,
    outputCol="features",
)


stages.append(
    assembler
)


# ============================================================
# TARGET LABEL
# ============================================================

label_indexer = StringIndexer(
    inputCol="Churn",
    outputCol="label",
    handleInvalid="keep",
)


stages.append(
    label_indexer
)


# ============================================================
# CREATE PIPELINE
# ============================================================

pipeline = Pipeline(
    stages=stages
)


print(
    "\nFitting preprocessing pipeline..."
)


pipeline_model = pipeline.fit(
    df
)


processed_df = pipeline_model.transform(
    df
)


print(
    "Pipeline transformation completed successfully."
)


# ============================================================
# CREATE FINAL DATASET
# ============================================================

final_df = processed_df.select(
    "features",
    "label",
)


print()
print("=" * 70)
print("PROCESSED DATASET")
print("=" * 70)


final_df.show(
    5,
    truncate=False,
)


# ============================================================
# VALIDATE PROCESSED DATA
# ============================================================

processed_rows = final_df.count()


first_row = final_df.first()


if first_row is None:

    spark.stop()

    raise ValueError(
        "Processed dataset is empty."
    )


feature_count = len(
    first_row["features"]
)


null_feature_rows = (
    final_df
    .filter(
        col("features").isNull()
    )
    .count()
)


null_label_rows = (
    final_df
    .filter(
        col("label").isNull()
    )
    .count()
)


print()
print("=" * 70)
print("PROCESSED DATA VALIDATION")
print("=" * 70)


print(
    "\nProcessed Rows:",
    processed_rows,
)

print(
    "Feature Count:",
    feature_count,
)

print(
    "Null Feature Rows:",
    null_feature_rows,
)

print(
    "Null Label Rows:",
    null_label_rows,
)


if null_feature_rows != 0:

    spark.stop()

    raise ValueError(
        "Processed dataset contains null features."
    )


if null_label_rows != 0:

    spark.stop()

    raise ValueError(
        "Processed dataset contains null labels."
    )


if processed_rows != clean_rows:

    spark.stop()

    raise ValueError(
        "Processed row count does not match "
        "the cleaned dataset row count."
    )


print(
    "\nProcessed dataset validation successful."
)


# ============================================================
# LABEL DISTRIBUTION
# ============================================================

print()
print("=" * 70)
print("LABEL DISTRIBUTION")
print("=" * 70)


final_df.groupBy(
    "label"
).count().orderBy(
    "label"
).show()


label_values = sorted(
    [
        row["label"]
        for row in (
            final_df
            .select(
                "label"
            )
            .distinct()
            .collect()
        )
    ]
)


print(
    "Unique Label Values:",
    label_values,
)


if not set(
    label_values
).issubset(
    {
        0.0,
        1.0,
    }
):

    spark.stop()

    raise ValueError(
        "Unexpected label values found: "
        f"{label_values}"
    )


# ============================================================
# FEATURE COUNT INFORMATION
# ============================================================

if feature_count == 45:

    print(
        "\nFeature count verified: 45"
    )

else:

    print(
        "\nWARNING:"
        " Expected 45 features based on the "
        "current preprocessing design, but found:",
        feature_count,
    )


# ============================================================
# SAVE PROCESSED DATA AS PARQUET
# ============================================================

print()
print("=" * 70)
print("SAVING PROCESSED DATA")
print("=" * 70)


print(
    "\nSaving to:"
)

print(
    OUTPUT_PATH
)


final_df.write \
    .mode(
        "overwrite"
    ) \
    .parquet(
        str(
            OUTPUT_PATH
        )
    )


print(
    "\nProcessed Parquet dataset saved successfully."
)


# ============================================================
# VERIFY SAVED PARQUET
# ============================================================

print()
print("=" * 70)
print("VERIFYING SAVED PARQUET DATA")
print("=" * 70)


verify_df = spark.read.parquet(
    str(
        OUTPUT_PATH
    )
)


verified_rows = verify_df.count()


verified_first_row = verify_df.first()


if verified_first_row is None:

    spark.stop()

    raise ValueError(
        "Saved Parquet dataset is empty."
    )


verified_feature_count = len(
    verified_first_row["features"]
)


print(
    "\nVerified Rows:",
    verified_rows,
)

print(
    "Verified Feature Count:",
    verified_feature_count,
)


print(
    "\nVerified Schema:"
)

verify_df.printSchema()


if verified_rows != processed_rows:

    spark.stop()

    raise ValueError(
        "Saved Parquet row count does not "
        "match processed dataset."
    )


if verified_feature_count != feature_count:

    spark.stop()

    raise ValueError(
        "Saved Parquet feature count does not "
        "match processed dataset."
    )


print(
    "\nPARQUET VERIFICATION SUCCESSFUL"
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print("PREPROCESSING SUMMARY")
print("=" * 70)


print(
    "\nOriginal Rows:",
    original_rows,
)

print(
    "Clean Rows:",
    clean_rows,
)

print(
    "Rows Removed:",
    removed_rows,
)

print(
    "Processed Rows:",
    processed_rows,
)

print(
    "Number of Features:",
    feature_count,
)

print(
    "Label Values:",
    label_values,
)

print(
    "\nOutput Dataset:"
)

print(
    OUTPUT_PATH
)


print()
print("=" * 70)
print("DATA PREPROCESSING COMPLETED SUCCESSFULLY")
print("=" * 70)


# ============================================================
# STOP SPARK
# ============================================================

spark.stop()