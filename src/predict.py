from pathlib import Path

from pyspark.sql import SparkSession

from pyspark.sql.functions import (
    col,
    when,
)

from pyspark.ml.functions import (
    vector_to_array,
)

from pyspark.ml.classification import (
    LogisticRegressionModel,
    DecisionTreeClassificationModel,
    RandomForestClassificationModel,
    GBTClassificationModel,
)


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_PATH = Path(__file__).resolve().parents[1]


# ============================================================
# SELECT MODEL
# ============================================================
#
# Available models:
#
# Logistic_Regression
# Decision_Tree
# Random_Forest
# Gradient_Boosted_Tree
#
# Logistic Regression is selected because it achieved
# the strongest overall baseline predictive performance.
# ============================================================

MODEL_NAME = "Logistic_Regression"


# ============================================================
# PATHS
# ============================================================

MODEL_PATH = (
    PROJECT_PATH
    / "models"
    / MODEL_NAME
)

DATA_PATH = (
    PROJECT_PATH
    / "data"
    / "processed"
    / "churn_data.parquet"
)

OUTPUT_PATH = (
    PROJECT_PATH
    / "results"
    / "churn_predictions"
)


# ============================================================
# START INFORMATION
# ============================================================

print()

print("=" * 70)

print(
    "CUSTOMER CHURN PREDICTION"
)

print("=" * 70)


print(
    "\nSelected Model:"
)

print(
    MODEL_NAME
)


print(
    "\nModel Path:"
)

print(
    MODEL_PATH
)


print(
    "\nData Path:"
)

print(
    DATA_PATH
)


print(
    "\nPrediction Output:"
)

print(
    OUTPUT_PATH
)


print("=" * 70)


# ============================================================
# CHECK PATHS
# ============================================================

if not MODEL_PATH.exists():

    raise FileNotFoundError(
        f"Model folder not found:\n"
        f"{MODEL_PATH}\n\n"
        f"Run train_model.py first."
    )


if not DATA_PATH.exists():

    raise FileNotFoundError(
        f"Processed Parquet data not found:\n"
        f"{DATA_PATH}\n\n"
        f"Run data_preprocessing.py first."
    )


OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# CREATE SPARK SESSION
# ============================================================

spark = (
    SparkSession.builder
    .appName(
        "Customer_Churn_Prediction"
    )
    .master(
        "local[2]"
    )
    .config(
        "spark.driver.memory",
        "4g"
    )
    .config(
        "spark.sql.shuffle.partitions",
        "2"
    )
    .config(
        "spark.default.parallelism",
        "2"
    )
    .config(
        "spark.hadoop.fs.defaultFS",
        "file:///"
    )
    .config(
        "spark.hadoop.fs.file.impl",
        "org.apache.hadoop.fs.RawLocalFileSystem"
    )
    .config(
        "spark.hadoop.mapreduce.fileoutputcommitter.algorithm.version",
        "2"
    )
    .getOrCreate()
)


spark.sparkContext.setLogLevel(
    "ERROR"
)


print()

print("=" * 70)

print(
    "SPARK STARTED SUCCESSFULLY"
)

print("=" * 70)


# ============================================================
# LOAD SELECTED MODEL
# ============================================================

print()

print("=" * 70)

print(
    "LOADING TRAINED MODEL"
)

print("=" * 70)


if MODEL_NAME == "Logistic_Regression":

    model = LogisticRegressionModel.load(
        str(
            MODEL_PATH
        )
    )


elif MODEL_NAME == "Decision_Tree":

    model = DecisionTreeClassificationModel.load(
        str(
            MODEL_PATH
        )
    )


elif MODEL_NAME == "Random_Forest":

    model = RandomForestClassificationModel.load(
        str(
            MODEL_PATH
        )
    )


elif MODEL_NAME == "Gradient_Boosted_Tree":

    model = GBTClassificationModel.load(
        str(
            MODEL_PATH
        )
    )


else:

    spark.stop()

    raise ValueError(
        f"Unknown MODEL_NAME: "
        f"{MODEL_NAME}"
    )


print(
    "MODEL LOADED SUCCESSFULLY"
)


print(
    "Model:",
    MODEL_NAME
)


# ============================================================
# OPTIONAL MODEL INFORMATION
# ============================================================

if MODEL_NAME == "Random_Forest":

    print(
        "Number of Trees:",
        model.getNumTrees
    )


if MODEL_NAME == "Gradient_Boosted_Tree":

    print(
        "Number of Trees:",
        model.getNumTrees
    )


# ============================================================
# LOAD PROCESSED PARQUET DATA
# ============================================================

print()

print("=" * 70)

print(
    "LOADING PROCESSED CUSTOMER DATA"
)

print("=" * 70)


df = spark.read.parquet(
    str(
        DATA_PATH
    )
)


total_rows = df.count()


print(
    "Total Rows:",
    total_rows
)


print()

print(
    "Dataset Schema:"
)


df.printSchema()


# ============================================================
# VALIDATE DATA
# ============================================================

print()

print("=" * 70)

print(
    "VALIDATING DATA"
)

print("=" * 70)


if "features" not in df.columns:

    spark.stop()

    raise ValueError(
        "Required 'features' column is missing."
    )


null_features = (
    df
    .filter(
        col(
            "features"
        ).isNull()
    )
    .count()
)


print(
    "Null Feature Rows:",
    null_features
)


if null_features > 0:

    spark.stop()

    raise ValueError(
        "Dataset contains null feature rows."
    )


first_row = df.first()


if first_row is None:

    spark.stop()

    raise ValueError(
        "Processed dataset is empty."
    )


feature_count = len(
    first_row[
        "features"
    ]
)


print(
    "Feature Count:",
    feature_count
)


print()

print(
    "Sample Processed Data:"
)


df.show(
    5,
    truncate=False
)


# ============================================================
# GENERATE PREDICTIONS
# ============================================================

print()

print("=" * 70)

print(
    "GENERATING PREDICTIONS"
)

print("=" * 70)


predictions = (
    model
    .transform(
        df
    )
    .cache()
)


prediction_count = (
    predictions.count()
)


print(
    "Predictions Generated:",
    prediction_count
)


# ============================================================
# VALIDATE PREDICTION OUTPUT
# ============================================================

required_prediction_columns = [
    "prediction",
    "probability",
]


for column_name in required_prediction_columns:

    if column_name not in predictions.columns:

        predictions.unpersist()

        spark.stop()

        raise ValueError(
            f"Prediction column missing: "
            f"{column_name}"
        )


# ============================================================
# SHOW RAW PREDICTIONS
# ============================================================

print()

print("=" * 70)

print(
    "RAW PREDICTIONS"
)

print("=" * 70)


display_columns = [
    "prediction",
    "probability",
]


if "label" in predictions.columns:

    display_columns.insert(
        0,
        "label",
    )


predictions.select(
    *display_columns
).show(
    20,
    truncate=False,
)


# ============================================================
# CONVERT PROBABILITY VECTOR TO ARRAY
# ============================================================

prediction_df = (
    predictions
    .withColumn(
        "probability_array",
        vector_to_array(
            col(
                "probability"
            )
        )
    )
)


# ============================================================
# CREATE FINAL OUTPUT
# ============================================================

if "label" in prediction_df.columns:

    final_output = prediction_df.select(

        col(
            "label"
        ).alias(
            "actual_label"
        ),

        col(
            "prediction"
        ).alias(
            "predicted_label"
        ),

        when(
            col(
                "prediction"
            )
            ==
            1.0,
            "Churn"
        ).otherwise(
            "No Churn"
        ).alias(
            "prediction_result"
        ),

        col(
            "probability_array"
        )[0].alias(
            "probability_no_churn"
        ),

        col(
            "probability_array"
        )[1].alias(
            "probability_churn"
        ),

    )


else:

    final_output = prediction_df.select(

        col(
            "prediction"
        ).alias(
            "predicted_label"
        ),

        when(
            col(
                "prediction"
            )
            ==
            1.0,
            "Churn"
        ).otherwise(
            "No Churn"
        ).alias(
            "prediction_result"
        ),

        col(
            "probability_array"
        )[0].alias(
            "probability_no_churn"
        ),

        col(
            "probability_array"
        )[1].alias(
            "probability_churn"
        ),

    )


# ============================================================
# SHOW FINAL OUTPUT
# ============================================================

print()

print("=" * 70)

print(
    "FINAL PREDICTION OUTPUT"
)

print("=" * 70)


final_output.show(
    20,
    truncate=False,
)


# ============================================================
# PREDICTION SUMMARY
# ============================================================

print()

print("=" * 70)

print(
    "PREDICTION SUMMARY"
)

print("=" * 70)


prediction_summary = (
    final_output
    .groupBy(
        "prediction_result"
    )
    .count()
    .orderBy(
        "prediction_result"
    )
)


prediction_summary.show()


predicted_churn = (
    final_output
    .filter(
        col(
            "predicted_label"
        )
        ==
        1.0
    )
    .count()
)


predicted_no_churn = (
    final_output
    .filter(
        col(
            "predicted_label"
        )
        ==
        0.0
    )
    .count()
)


print(
    "Total Customers:",
    prediction_count
)


print(
    "Predicted No Churn:",
    predicted_no_churn
)


print(
    "Predicted Churn:",
    predicted_churn
)


if prediction_count > 0:

    churn_percentage = (
        predicted_churn
        /
        prediction_count
        *
        100
    )

else:

    churn_percentage = 0.0


print(
    "Predicted Churn Percentage:",
    round(
        churn_percentage,
        2
    ),
    "%"
)


# ============================================================
# SAVE PREDICTIONS
# ============================================================

print()

print("=" * 70)

print(
    "SAVING PREDICTIONS"
)

print("=" * 70)


final_output.write \
    .mode(
        "overwrite"
    ) \
    .option(
        "header",
        "true"
    ) \
    .csv(
        str(
            OUTPUT_PATH
        )
    )


print()

print(
    "PREDICTIONS SAVED SUCCESSFULLY"
)


print(
    OUTPUT_PATH
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print()

print("=" * 70)

print(
    "PREDICTION PIPELINE COMPLETE"
)

print("=" * 70)


print(
    "Model Used:",
    MODEL_NAME
)


print(
    "Input Rows:",
    total_rows
)


print(
    "Predictions:",
    prediction_count
)


print(
    "Feature Count:",
    feature_count
)


print(
    "Predicted No Churn:",
    predicted_no_churn
)


print(
    "Predicted Churn:",
    predicted_churn
)


print(
    "Output:"
)


print(
    OUTPUT_PATH
)


# ============================================================
# CLEANUP
# ============================================================

predictions.unpersist()


spark.stop()


print()

print("=" * 70)

print(
    "SPARK STOPPED"
)

print("=" * 70)