import os
import shutil
import csv
import time
from pathlib import Path


# ============================================================
# IMPORTS
# ============================================================

from pyspark.sql import SparkSession

from pyspark.sql.functions import (
    col
)

from pyspark.ml.classification import (
    LogisticRegression,
    DecisionTreeClassifier,
    RandomForestClassifier,
    GBTClassifier
)

from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator
)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_PATH = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_PATH
    / "data"
    / "processed"
    / "churn_data.parquet"
)

MODEL_PATH = (
    PROJECT_PATH
    / "models"
)

RESULTS_PATH = (
    PROJECT_PATH
    / "results"
)

COMPARISON_FILE = (
    RESULTS_PATH
    / "model_comparison.csv"
)


# ============================================================
# CREATE DIRECTORIES
# ============================================================

MODEL_PATH.mkdir(
    parents=True,
    exist_ok=True
)

RESULTS_PATH.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# START MESSAGE
# ============================================================

print()

print("=" * 70)

print(
    "SCALABLE CUSTOMER CHURN MODEL TRAINING"
)

print("=" * 70)


print(
    "\nDATA:"
)

print(
    DATA_PATH
)


print(
    "\nMODELS:"
)

print(
    MODEL_PATH
)


print(
    "\nRESULTS:"
)

print(
    COMPARISON_FILE
)


print("=" * 70)


# ============================================================
# CHECK PROCESSED DATA
# ============================================================

if not DATA_PATH.exists():

    raise FileNotFoundError(
        f"Processed dataset not found:\n"
        f"{DATA_PATH}\n\n"
        f"Run data_preprocessing.py first."
    )


# ============================================================
# CREATE SPARK SESSION
# ============================================================

spark = (
    SparkSession.builder
    .appName(
        "Telco_Churn_Model_Training"
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

print(
    "SPARK STARTED SUCCESSFULLY"
)


# ============================================================
# LOAD PARQUET DATA
# ============================================================

print()

print("=" * 70)

print(
    "LOADING PROCESSED PARQUET DATA"
)

print("=" * 70)


df = spark.read.parquet(
    str(
        DATA_PATH
    )
)


print()

print(
    "Original Parquet Schema:"
)


df.printSchema()


# ============================================================
# IMPORTANT FIX:
# REMOVE STRINGINDEXER LABEL METADATA
# ============================================================

print()

print("=" * 70)

print(
    "CLEANING BINARY LABEL METADATA"
)

print("=" * 70)


# The label was originally produced by StringIndexer.
#
# Because handleInvalid="keep" was used during preprocessing,
# Spark metadata may describe an additional possible class even
# though the actual dataset contains only labels 0 and 1.
#
# Rebuilding the label as a plain double removes that metadata
# and ensures that Spark treats this as binary classification.

df = (
    df
    .withColumn(
        "binary_label",
        col(
            "label"
        ).cast(
            "double"
        )
    )
    .drop(
        "label"
    )
    .withColumnRenamed(
        "binary_label",
        "label"
    )
)


print(
    "Binary label column rebuilt successfully."
)


# ============================================================
# DATASET VALIDATION
# ============================================================

print()

print("=" * 70)

print(
    "VALIDATING DATASET"
)

print("=" * 70)


required_columns = [
    "features",
    "label"
]


for required_column in required_columns:

    if required_column not in df.columns:

        spark.stop()

        raise ValueError(
            f"Missing required column: "
            f"{required_column}"
        )


total_rows = df.count()


null_feature_rows = (
    df
    .filter(
        col(
            "features"
        ).isNull()
    )
    .count()
)


null_label_rows = (
    df
    .filter(
        col(
            "label"
        ).isNull()
    )
    .count()
)


print(
    "Total Rows:",
    total_rows
)


print(
    "Null Feature Rows:",
    null_feature_rows
)


print(
    "Null Label Rows:",
    null_label_rows
)


if null_feature_rows > 0:

    spark.stop()

    raise ValueError(
        "Dataset contains null feature rows."
    )


if null_label_rows > 0:

    spark.stop()

    raise ValueError(
        "Dataset contains null label rows."
    )


# ============================================================
# FEATURE COUNT
# ============================================================

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


# ============================================================
# CHECK ACTUAL LABEL VALUES
# ============================================================

print()

print(
    "Label Distribution:"
)


df.groupBy(
    "label"
).count().orderBy(
    "label"
).show()


label_values = sorted(
    [
        row["label"]

        for row in (
            df
            .select(
                "label"
            )
            .distinct()
            .collect()
        )
    ]
)


print(
    "Unique Labels:",
    label_values
)


if label_values != [
    0.0,
    1.0
]:

    spark.stop()

    raise ValueError(
        "Expected binary labels [0.0, 1.0], "
        f"but found {label_values}"
    )


print(
    "BINARY LABEL VALIDATION SUCCESSFUL"
)


# ============================================================
# DISPLAY FINAL SCHEMA
# ============================================================

print()

print(
    "Final Training Schema:"
)


df.printSchema()


print()

print(
    "Sample Data:"
)


df.show(
    5,
    truncate=False
)


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

print()

print("=" * 70)

print(
    "TRAIN / TEST SPLIT"
)

print("=" * 70)


train_data, test_data = (
    df.randomSplit(
        [
            0.8,
            0.2
        ],
        seed=42
    )
)


# Cache because all four models use the same split.

train_data = (
    train_data.cache()
)

test_data = (
    test_data.cache()
)


train_rows = (
    train_data.count()
)

test_rows = (
    test_data.count()
)


print(
    "Total Dataset Rows:",
    total_rows
)


print(
    "Training Rows:",
    train_rows
)


print(
    "Testing Rows:",
    test_rows
)


training_percentage = (
    train_rows
    /
    total_rows
    *
    100
)


testing_percentage = (
    test_rows
    /
    total_rows
    *
    100
)


print(
    "Training Percentage:",
    round(
        training_percentage,
        2
    ),
    "%"
)


print(
    "Testing Percentage:",
    round(
        testing_percentage,
        2
    ),
    "%"
)


# ============================================================
# EVALUATORS
# ============================================================

accuracy_evaluator = (
    MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="accuracy"
    )
)


weighted_f1_evaluator = (
    MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="f1"
    )
)


auc_evaluator = (
    BinaryClassificationEvaluator(
        labelCol="label",
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC"
    )
)


# ============================================================
# RESULTS STORAGE
# ============================================================

experiment_results = []


# ============================================================
# TRAIN + EVALUATE FUNCTION
# ============================================================

def train_and_save(
    model,
    model_name
):

    print()

    print("=" * 70)

    print(
        "TRAINING:",
        model_name
    )

    print("=" * 70)


    # --------------------------------------------------------
    # TRAINING TIME
    # --------------------------------------------------------

    training_start = (
        time.perf_counter()
    )


    trained_model = model.fit(
        train_data
    )


    training_end = (
        time.perf_counter()
    )


    training_time = (
        training_end
        -
        training_start
    )


    # --------------------------------------------------------
    # PREDICTION TIME
    # --------------------------------------------------------

    prediction_start = (
        time.perf_counter()
    )


    predictions = (
        trained_model
        .transform(
            test_data
        )
        .cache()
    )


    # count() forces Spark to execute the lazy prediction
    # operation, allowing prediction time to be measured.

    prediction_count = (
        predictions.count()
    )


    prediction_end = (
        time.perf_counter()
    )


    prediction_time = (
        prediction_end
        -
        prediction_start
    )


    # --------------------------------------------------------
    # VERIFY RAW PREDICTION VECTOR
    # --------------------------------------------------------

    raw_prediction_sample = (
        predictions
        .select(
            "rawPrediction"
        )
        .first()
    )


    if raw_prediction_sample is not None:

        raw_vector_length = len(
            raw_prediction_sample[
                "rawPrediction"
            ]
        )

    else:

        raw_vector_length = 0


    print(
        "\nRaw Prediction Vector Length:",
        raw_vector_length
    )


    if raw_vector_length != 2:

        predictions.unpersist()

        raise ValueError(
            "Binary classifier should produce "
            "rawPrediction vectors of length 2, "
            f"but got {raw_vector_length}."
        )


    # --------------------------------------------------------
    # MODEL METRICS
    # --------------------------------------------------------

    accuracy = (
        accuracy_evaluator.evaluate(
            predictions
        )
    )


    weighted_f1 = (
        weighted_f1_evaluator.evaluate(
            predictions
        )
    )


    auc = (
        auc_evaluator.evaluate(
            predictions
        )
    )


    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    tn = (
        predictions
        .filter(
            (
                col(
                    "label"
                )
                ==
                0.0
            )
            &
            (
                col(
                    "prediction"
                )
                ==
                0.0
            )
        )
        .count()
    )


    fp = (
        predictions
        .filter(
            (
                col(
                    "label"
                )
                ==
                0.0
            )
            &
            (
                col(
                    "prediction"
                )
                ==
                1.0
            )
        )
        .count()
    )


    fn = (
        predictions
        .filter(
            (
                col(
                    "label"
                )
                ==
                1.0
            )
            &
            (
                col(
                    "prediction"
                )
                ==
                0.0
            )
        )
        .count()
    )


    tp = (
        predictions
        .filter(
            (
                col(
                    "label"
                )
                ==
                1.0
            )
            &
            (
                col(
                    "prediction"
                )
                ==
                1.0
            )
        )
        .count()
    )


    # --------------------------------------------------------
    # CHURN PRECISION
    # --------------------------------------------------------

    if (
        tp
        +
        fp
    ) > 0:

        churn_precision = (
            tp
            /
            (
                tp
                +
                fp
            )
        )

    else:

        churn_precision = 0.0


    # --------------------------------------------------------
    # CHURN RECALL
    # --------------------------------------------------------

    if (
        tp
        +
        fn
    ) > 0:

        churn_recall = (
            tp
            /
            (
                tp
                +
                fn
            )
        )

    else:

        churn_recall = 0.0


    # --------------------------------------------------------
    # CHURN F1
    # --------------------------------------------------------

    if (
        churn_precision
        +
        churn_recall
    ) > 0:

        churn_f1 = (
            2
            *
            churn_precision
            *
            churn_recall
            /
            (
                churn_precision
                +
                churn_recall
            )
        )

    else:

        churn_f1 = 0.0


    # --------------------------------------------------------
    # PREDICTION THROUGHPUT
    # --------------------------------------------------------

    if prediction_time > 0:

        prediction_throughput = (
            prediction_count
            /
            prediction_time
        )

    else:

        prediction_throughput = 0.0


    # --------------------------------------------------------
    # DISPLAY PREDICTIVE PERFORMANCE
    # --------------------------------------------------------

    print()

    print(
        "MODEL PERFORMANCE"
    )

    print(
        "-" * 70
    )


    print(
        "Accuracy:",
        round(
            accuracy,
            4
        )
    )


    print(
        "Precision (Churn):",
        round(
            churn_precision,
            4
        )
    )


    print(
        "Recall (Churn):",
        round(
            churn_recall,
            4
        )
    )


    print(
        "F1 (Churn):",
        round(
            churn_f1,
            4
        )
    )


    print(
        "Weighted F1:",
        round(
            weighted_f1,
            4
        )
    )


    print(
        "ROC-AUC:",
        round(
            auc,
            4
        )
    )


    # --------------------------------------------------------
    # DISPLAY CONFUSION MATRIX
    # --------------------------------------------------------

    print()

    print(
        "CONFUSION MATRIX"
    )

    print(
        "-" * 70
    )


    print(
        "True Negatives :",
        tn
    )


    print(
        "False Positives:",
        fp
    )


    print(
        "False Negatives:",
        fn
    )


    print(
        "True Positives :",
        tp
    )


    # --------------------------------------------------------
    # DISPLAY COMPUTATIONAL PERFORMANCE
    # --------------------------------------------------------

    print()

    print(
        "COMPUTATIONAL PERFORMANCE"
    )

    print(
        "-" * 70
    )


    print(
        "Training Time:",
        round(
            training_time,
            4
        ),
        "seconds"
    )


    print(
        "Prediction Time:",
        round(
            prediction_time,
            4
        ),
        "seconds"
    )


    print(
        "Prediction Throughput:",
        round(
            prediction_throughput,
            2
        ),
        "rows/second"
    )


    # --------------------------------------------------------
    # SAVE MODEL
    # --------------------------------------------------------

    model_save_path = (
        MODEL_PATH
        /
        model_name
    )


    print()

    print(
        "Saving Model:"
    )

    print(
        model_save_path
    )


    if model_save_path.exists():

        shutil.rmtree(
            model_save_path
        )


    trained_model.write() \
        .overwrite() \
        .save(
            str(
                model_save_path
            )
        )


    print(
        "MODEL SAVED SUCCESSFULLY"
    )


    # --------------------------------------------------------
    # STORE RESULT
    # --------------------------------------------------------

    result = {

        "Model":
            model_name,

        "Dataset_Rows":
            total_rows,

        "Training_Rows":
            train_rows,

        "Testing_Rows":
            test_rows,

        "Feature_Count":
            feature_count,

        "Accuracy":
            round(
                accuracy,
                6
            ),

        "Precision_Churn":
            round(
                churn_precision,
                6
            ),

        "Recall_Churn":
            round(
                churn_recall,
                6
            ),

        "F1_Churn":
            round(
                churn_f1,
                6
            ),

        "Weighted_F1":
            round(
                weighted_f1,
                6
            ),

        "ROC_AUC":
            round(
                auc,
                6
            ),

        "True_Negatives":
            tn,

        "False_Positives":
            fp,

        "False_Negatives":
            fn,

        "True_Positives":
            tp,

        "Training_Time_Seconds":
            round(
                training_time,
                6
            ),

        "Prediction_Time_Seconds":
            round(
                prediction_time,
                6
            ),

        "Prediction_Throughput_Rows_Per_Second":
            round(
                prediction_throughput,
                2
            )

    }


    experiment_results.append(
        result
    )


    predictions.unpersist()


# ============================================================
# DEFINE MODELS
# ============================================================

models = [

    (
        LogisticRegression(
            featuresCol="features",
            labelCol="label",
            maxIter=50
        ),

        "Logistic_Regression"
    ),

    (
        DecisionTreeClassifier(
            featuresCol="features",
            labelCol="label",
            maxDepth=5,
            seed=42
        ),

        "Decision_Tree"
    ),

    (
        RandomForestClassifier(
            featuresCol="features",
            labelCol="label",
            numTrees=50,
            maxDepth=8,
            seed=42
        ),

        "Random_Forest"
    ),

    (
        GBTClassifier(
            featuresCol="features",
            labelCol="label",
            maxIter=30,
            seed=42
        ),

        "Gradient_Boosted_Tree"
    )

]


# ============================================================
# TRAIN ALL MODELS
# ============================================================

for model, model_name in models:

    train_and_save(
        model,
        model_name
    )


# ============================================================
# SAVE RESULTS TO CSV
# ============================================================

print()

print("=" * 70)

print(
    "SAVING MODEL COMPARISON RESULTS"
)

print("=" * 70)


fieldnames = [

    "Model",

    "Dataset_Rows",
    "Training_Rows",
    "Testing_Rows",

    "Feature_Count",

    "Accuracy",

    "Precision_Churn",
    "Recall_Churn",
    "F1_Churn",

    "Weighted_F1",

    "ROC_AUC",

    "True_Negatives",
    "False_Positives",
    "False_Negatives",
    "True_Positives",

    "Training_Time_Seconds",

    "Prediction_Time_Seconds",

    "Prediction_Throughput_Rows_Per_Second"

]


with open(
    COMPARISON_FILE,
    "w",
    newline="",
    encoding="utf-8"
) as csv_file:

    writer = csv.DictWriter(
        csv_file,
        fieldnames=fieldnames
    )


    writer.writeheader()


    writer.writerows(
        experiment_results
    )


print()

print(
    "MODEL COMPARISON SAVED SUCCESSFULLY"
)


print(
    COMPARISON_FILE
)


# ============================================================
# FINAL MODEL COMPARISON
# ============================================================

print()

print(
    "=" * 125
)

print(
    "FINAL MODEL COMPARISON"
)

print(
    "=" * 125
)


print(
    f"{'Model':<25}"
    f"{'Accuracy':<11}"
    f"{'Precision':<11}"
    f"{'Recall':<11}"
    f"{'Churn F1':<11}"
    f"{'Weighted F1':<13}"
    f"{'AUC':<10}"
    f"{'Train(s)':<11}"
    f"{'Predict(s)':<11}"
)


print(
    "-" * 125
)


for result in experiment_results:

    print(

        f"{result['Model']:<25}"

        f"{result['Accuracy']:<11.4f}"

        f"{result['Precision_Churn']:<11.4f}"

        f"{result['Recall_Churn']:<11.4f}"

        f"{result['F1_Churn']:<11.4f}"

        f"{result['Weighted_F1']:<13.4f}"

        f"{result['ROC_AUC']:<10.4f}"

        f"{result['Training_Time_Seconds']:<11.4f}"

        f"{result['Prediction_Time_Seconds']:<11.4f}"

    )


print(
    "=" * 125
)


# ============================================================
# BEST MODEL SUMMARY
# ============================================================

if experiment_results:

    best_accuracy_model = max(
        experiment_results,
        key=lambda x: x[
            "Accuracy"
        ]
    )


    best_auc_model = max(
        experiment_results,
        key=lambda x: x[
            "ROC_AUC"
        ]
    )


    best_churn_f1_model = max(
        experiment_results,
        key=lambda x: x[
            "F1_Churn"
        ]
    )


    print()

    print("=" * 70)

    print(
        "BEST MODEL SUMMARY"
    )

    print("=" * 70)


    print(
        "Best Accuracy:",
        best_accuracy_model[
            "Model"
        ],
        "-",
        best_accuracy_model[
            "Accuracy"
        ]
    )


    print(
        "Best ROC-AUC:",
        best_auc_model[
            "Model"
        ],
        "-",
        best_auc_model[
            "ROC_AUC"
        ]
    )


    print(
        "Best Churn F1:",
        best_churn_f1_model[
            "Model"
        ],
        "-",
        best_churn_f1_model[
            "F1_Churn"
        ]
    )


# ============================================================
# CLEANUP
# ============================================================

train_data.unpersist()

test_data.unpersist()


spark.stop()


print()

print("=" * 70)

print(
    "ALL MODELS TRAINED AND EVALUATED"
)

print("=" * 70)


print(
    "\nResults saved:"
)


print(
    COMPARISON_FILE
)


print(
    "\nSPARK STOPPED"
)


print("=" * 70)