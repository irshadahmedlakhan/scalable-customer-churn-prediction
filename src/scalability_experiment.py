import csv
import gc
import statistics
import time
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql.functions import col

from pyspark.ml.classification import (
    LogisticRegression,
    DecisionTreeClassifier,
    RandomForestClassifier,
    GBTClassifier,
)

from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator,
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

RESULTS_PATH = (
    PROJECT_PATH
    / "results"
)

TRIAL_RESULTS_FILE = (
    RESULTS_PATH
    / "scalability_trials.csv"
)

SUMMARY_RESULTS_FILE = (
    RESULTS_PATH
    / "scalability_repeated_results.csv"
)


# Temporary checkpoint files.
# These allow a crashed experiment to continue later without
# overwriting the existing completed research results.

TRIAL_CHECKPOINT_FILE = (
    RESULTS_PATH
    / "scalability_trials_checkpoint.csv"
)

SUMMARY_CHECKPOINT_FILE = (
    RESULTS_PATH
    / "scalability_summary_checkpoint.csv"
)


RESULTS_PATH.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# EXPERIMENT CONFIGURATION
# ============================================================

SCALE_FACTORS = [
    1,
    5,
    10,
    20,
    50,
    100,
]

NUM_TRIALS = 3

NUM_PARTITIONS = 2


MODEL_NAMES = [
    "Logistic_Regression",
    "Decision_Tree",
    "Random_Forest",
    "Gradient_Boosted_Tree",
]


# ============================================================
# CSV FIELD DEFINITIONS
# ============================================================

TRIAL_CHECKPOINT_FIELDS = [
    "Scale_Factor",
    "Model",
    "Trial",
    "Base_Dataset_Rows",
    "Synthetic_Total_Rows",
    "Training_Rows",
    "Testing_Rows",
    "Feature_Count",
    "Training_Time_Seconds",
    "Prediction_Time_Seconds",
    "Training_Throughput_Rows_Per_Second",
    "Prediction_Throughput_Rows_Per_Second",

    # Predictive metrics are stored for Trial 1 so that
    # they survive a JVM crash and can be reused on resume.
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
]


TRIAL_FINAL_FIELDS = [
    "Scale_Factor",
    "Model",
    "Trial",
    "Base_Dataset_Rows",
    "Synthetic_Total_Rows",
    "Training_Rows",
    "Testing_Rows",
    "Feature_Count",
    "Training_Time_Seconds",
    "Prediction_Time_Seconds",
    "Training_Throughput_Rows_Per_Second",
    "Prediction_Throughput_Rows_Per_Second",
]


SUMMARY_FIELDS = [
    "Scale_Factor",
    "Model",
    "Trials",
    "Base_Dataset_Rows",
    "Synthetic_Total_Rows",
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
    "Training_Time_Run_1",
    "Training_Time_Run_2",
    "Training_Time_Run_3",
    "Mean_Training_Time_Seconds",
    "Std_Training_Time_Seconds",
    "Prediction_Time_Run_1",
    "Prediction_Time_Run_2",
    "Prediction_Time_Run_3",
    "Mean_Prediction_Time_Seconds",
    "Std_Prediction_Time_Seconds",
    "Mean_Training_Throughput_Rows_Per_Second",
    "Std_Training_Throughput_Rows_Per_Second",
    "Mean_Prediction_Throughput_Rows_Per_Second",
    "Std_Prediction_Throughput_Rows_Per_Second",
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def calculate_mean(values):

    return statistics.mean(values)


def calculate_std(values):

    if len(values) <= 1:
        return 0.0

    return statistics.stdev(values)


def save_csv(
    file_path,
    rows,
    fieldnames,
):

    with open(
        file_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for row in rows:

            filtered_row = {
                field:
                    row.get(
                        field,
                        ""
                    )
                for field in fieldnames
            }

            writer.writerow(
                filtered_row
            )


def load_csv(
    file_path
):

    if not file_path.exists():

        return []


    with open(
        file_path,
        "r",
        newline="",
        encoding="utf-8",
    ) as csv_file:

        return list(
            csv.DictReader(
                csv_file
            )
        )


def normalize_trial_row(
    row
):

    return {
        "Scale_Factor":
            int(
                row["Scale_Factor"]
            ),

        "Model":
            row["Model"],

        "Trial":
            int(
                row["Trial"]
            ),

        "Base_Dataset_Rows":
            int(
                row["Base_Dataset_Rows"]
            ),

        "Synthetic_Total_Rows":
            int(
                row["Synthetic_Total_Rows"]
            ),

        "Training_Rows":
            int(
                row["Training_Rows"]
            ),

        "Testing_Rows":
            int(
                row["Testing_Rows"]
            ),

        "Feature_Count":
            int(
                row["Feature_Count"]
            ),

        "Training_Time_Seconds":
            float(
                row["Training_Time_Seconds"]
            ),

        "Prediction_Time_Seconds":
            float(
                row["Prediction_Time_Seconds"]
            ),

        "Training_Throughput_Rows_Per_Second":
            float(
                row[
                    "Training_Throughput_Rows_Per_Second"
                ]
            ),

        "Prediction_Throughput_Rows_Per_Second":
            float(
                row[
                    "Prediction_Throughput_Rows_Per_Second"
                ]
            ),

        "Accuracy":
            row.get(
                "Accuracy",
                ""
            ),

        "Precision_Churn":
            row.get(
                "Precision_Churn",
                ""
            ),

        "Recall_Churn":
            row.get(
                "Recall_Churn",
                ""
            ),

        "F1_Churn":
            row.get(
                "F1_Churn",
                ""
            ),

        "Weighted_F1":
            row.get(
                "Weighted_F1",
                ""
            ),

        "ROC_AUC":
            row.get(
                "ROC_AUC",
                ""
            ),

        "True_Negatives":
            row.get(
                "True_Negatives",
                ""
            ),

        "False_Positives":
            row.get(
                "False_Positives",
                ""
            ),

        "False_Negatives":
            row.get(
                "False_Negatives",
                ""
            ),

        "True_Positives":
            row.get(
                "True_Positives",
                ""
            ),
    }


def normalize_summary_row(
    row
):

    normalized = dict(row)

    integer_fields = [
        "Scale_Factor",
        "Trials",
        "Base_Dataset_Rows",
        "Synthetic_Total_Rows",
        "Training_Rows",
        "Testing_Rows",
        "Feature_Count",
        "True_Negatives",
        "False_Positives",
        "False_Negatives",
        "True_Positives",
    ]


    float_fields = [
        "Accuracy",
        "Precision_Churn",
        "Recall_Churn",
        "F1_Churn",
        "Weighted_F1",
        "ROC_AUC",
        "Training_Time_Run_1",
        "Training_Time_Run_2",
        "Training_Time_Run_3",
        "Mean_Training_Time_Seconds",
        "Std_Training_Time_Seconds",
        "Prediction_Time_Run_1",
        "Prediction_Time_Run_2",
        "Prediction_Time_Run_3",
        "Mean_Prediction_Time_Seconds",
        "Std_Prediction_Time_Seconds",
        "Mean_Training_Throughput_Rows_Per_Second",
        "Std_Training_Throughput_Rows_Per_Second",
        "Mean_Prediction_Throughput_Rows_Per_Second",
        "Std_Prediction_Throughput_Rows_Per_Second",
    ]


    for field in integer_fields:

        normalized[field] = int(
            float(
                normalized[field]
            )
        )


    for field in float_fields:

        normalized[field] = float(
            normalized[field]
        )


    return normalized


def trial_key(
    scale_factor,
    model_name,
    trial_number,
):

    return (
        int(
            scale_factor
        ),
        model_name,
        int(
            trial_number
        ),
    )


def summary_key(
    scale_factor,
    model_name,
):

    return (
        int(
            scale_factor
        ),
        model_name,
    )


# ============================================================
# START MESSAGE
# ============================================================

print()

print("=" * 90)

print(
    "SCALABLE CUSTOMER CHURN - RESUMABLE SCALABILITY EXPERIMENT"
)

print("=" * 90)


print(
    "\nInput Data:"
)

print(
    DATA_PATH
)


print(
    "\nFinal Raw Trial Results:"
)

print(
    TRIAL_RESULTS_FILE
)


print(
    "\nFinal Repeated Summary Results:"
)

print(
    SUMMARY_RESULTS_FILE
)


print(
    "\nTrial Checkpoint:"
)

print(
    TRIAL_CHECKPOINT_FILE
)


print(
    "\nSummary Checkpoint:"
)

print(
    SUMMARY_CHECKPOINT_FILE
)


print(
    "\nScale Factors:"
)

print(
    SCALE_FACTORS
)


print(
    "\nMeasured Trials Per Model / Scale:"
)

print(
    NUM_TRIALS
)


print(
    "\nSpark Local Threads:"
)

print(
    2
)


print("=" * 90)


# ============================================================
# CHECK INPUT DATA
# ============================================================

if not DATA_PATH.exists():

    raise FileNotFoundError(
        f"Processed dataset not found:\n"
        f"{DATA_PATH}\n\n"
        f"Run data_preprocessing.py first."
    )


# ============================================================
# LOAD CHECKPOINTS
# ============================================================

print()

print("=" * 90)

print(
    "CHECKING FOR PREVIOUS INCOMPLETE EXPERIMENT"
)

print("=" * 90)


raw_trial_checkpoint_rows = (
    load_csv(
        TRIAL_CHECKPOINT_FILE
    )
)


raw_summary_checkpoint_rows = (
    load_csv(
        SUMMARY_CHECKPOINT_FILE
    )
)


trial_results = [
    normalize_trial_row(
        row
    )
    for row in raw_trial_checkpoint_rows
]


summary_results = [
    normalize_summary_row(
        row
    )
    for row in raw_summary_checkpoint_rows
]


completed_trial_keys = {
    trial_key(
        row["Scale_Factor"],
        row["Model"],
        row["Trial"],
    )
    for row in trial_results
}


completed_summary_keys = {
    summary_key(
        row["Scale_Factor"],
        row["Model"],
    )
    for row in summary_results
}


if trial_results or summary_results:

    print(
        "CHECKPOINT FOUND"
    )

    print(
        "Completed Trials:",
        len(
            trial_results
        )
    )

    print(
        "Completed Model/Scale Summaries:",
        len(
            summary_results
        )
    )

    print(
        "\nThe experiment will resume from the checkpoint."
    )


else:

    print(
        "No checkpoint found."
    )

    print(
        "Starting a new scalability experiment."
    )


# ============================================================
# CREATE SPARK SESSION
# ============================================================

spark = (
    SparkSession.builder
    .appName(
        "Telco_Churn_Resumable_Scalability_Experiment"
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
        str(
            NUM_PARTITIONS
        )
    )
    .config(
        "spark.default.parallelism",
        str(
            NUM_PARTITIONS
        )
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

print("=" * 90)

print(
    "SPARK STARTED SUCCESSFULLY"
)

print("=" * 90)


# ============================================================
# LOAD BASE DATASET
# ============================================================

print()

print("=" * 90)

print(
    "LOADING BASE DATASET"
)

print("=" * 90)


df = spark.read.parquet(
    str(
        DATA_PATH
    )
)


# ============================================================
# REMOVE OLD LABEL METADATA
# ============================================================

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


# ============================================================
# VALIDATE BASE DATA
# ============================================================

base_total_rows = (
    df.count()
)


first_row = (
    df.first()
)


if first_row is None:

    spark.stop()

    raise ValueError(
        "Dataset is empty."
    )


feature_count = len(
    first_row[
        "features"
    ]
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


null_labels = (
    df
    .filter(
        col(
            "label"
        ).isNull()
    )
    .count()
)


unique_labels = sorted(
    [
        row[
            "label"
        ]

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
    "\nBase Dataset Rows:",
    base_total_rows
)


print(
    "Feature Count:",
    feature_count
)


print(
    "Null Feature Rows:",
    null_features
)


print(
    "Null Label Rows:",
    null_labels
)


print(
    "Unique Labels:",
    unique_labels
)


print(
    "\nBase Label Distribution:"
)


(
    df
    .groupBy(
        "label"
    )
    .count()
    .orderBy(
        "label"
    )
    .show()
)


if null_features != 0:

    spark.stop()

    raise ValueError(
        "Null features were found."
    )


if null_labels != 0:

    spark.stop()

    raise ValueError(
        "Null labels were found."
    )


if unique_labels != [
    0.0,
    1.0,
]:

    spark.stop()

    raise ValueError(
        "Expected binary labels [0.0, 1.0], "
        f"but found {unique_labels}"
    )


print(
    "BASE DATA VALIDATION SUCCESSFUL"
)


# ============================================================
# CREATE FIXED BASE TRAIN / TEST SPLIT
# ============================================================

print()

print("=" * 90)

print(
    "CREATING FIXED BASE TRAIN / TEST SPLIT"
)

print("=" * 90)


base_train, base_test = (
    df.randomSplit(
        [
            0.8,
            0.2,
        ],
        seed=42,
    )
)


base_train = (
    base_train
    .repartition(
        NUM_PARTITIONS
    )
    .cache()
)


base_test = (
    base_test
    .repartition(
        NUM_PARTITIONS
    )
    .cache()
)


base_train_rows = (
    base_train.count()
)


base_test_rows = (
    base_test.count()
)


print(
    "Base Training Rows:",
    base_train_rows
)


print(
    "Base Testing Rows:",
    base_test_rows
)


print(
    "Total:",
    base_train_rows
    +
    base_test_rows
)


# ============================================================
# EVALUATORS
# ============================================================

accuracy_evaluator = (
    MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="accuracy",
    )
)


weighted_f1_evaluator = (
    MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="f1",
    )
)


auc_evaluator = (
    BinaryClassificationEvaluator(
        labelCol="label",
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC",
    )
)


# ============================================================
# MODEL FACTORY
# ============================================================

def create_model(
    model_name
):

    if model_name == "Logistic_Regression":

        return LogisticRegression(
            featuresCol="features",
            labelCol="label",
            maxIter=50,
        )


    if model_name == "Decision_Tree":

        return DecisionTreeClassifier(
            featuresCol="features",
            labelCol="label",
            maxDepth=5,
            seed=42,
        )


    if model_name == "Random_Forest":

        return RandomForestClassifier(
            featuresCol="features",
            labelCol="label",
            numTrees=50,
            maxDepth=8,
            seed=42,
        )


    if model_name == "Gradient_Boosted_Tree":

        return GBTClassifier(
            featuresCol="features",
            labelCol="label",
            maxIter=30,
            seed=42,
        )


    raise ValueError(
        f"Unknown model: "
        f"{model_name}"
    )


# ============================================================
# CONFUSION MATRIX / CHURN METRICS
# ============================================================

def calculate_confusion_metrics(
    predictions
):

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


    if (
        tp
        +
        fp
    ) > 0:

        precision_churn = (
            tp
            /
            (
                tp
                +
                fp
            )
        )

    else:

        precision_churn = 0.0


    if (
        tp
        +
        fn
    ) > 0:

        recall_churn = (
            tp
            /
            (
                tp
                +
                fn
            )
        )

    else:

        recall_churn = 0.0


    if (
        precision_churn
        +
        recall_churn
    ) > 0:

        f1_churn = (
            2
            *
            precision_churn
            *
            recall_churn
            /
            (
                precision_churn
                +
                recall_churn
            )
        )

    else:

        f1_churn = 0.0


    return {
        "True_Negatives":
            tn,

        "False_Positives":
            fp,

        "False_Negatives":
            fn,

        "True_Positives":
            tp,

        "Precision_Churn":
            precision_churn,

        "Recall_Churn":
            recall_churn,

        "F1_Churn":
            f1_churn,
    }


# ============================================================
# GLOBAL SPARK / JVM WARM-UP
# ============================================================

print()

print("=" * 90)

print(
    "GLOBAL SPARK / JVM WARM-UP"
)

print("=" * 90)


warmup_value = (
    spark
    .range(
        10000
    )
    .repartition(
        NUM_PARTITIONS
    )
    .count()
)


print(
    "Basic Spark Warm-Up Rows:",
    warmup_value
)


# ============================================================
# ALGORITHM WARM-UP
# ============================================================

print()

print("=" * 90)

print(
    "ALGORITHM WARM-UP"
)

print("=" * 90)


for model_name in MODEL_NAMES:

    print()

    print(
        f"Warming Up: "
        f"{model_name}"
    )


    warmup_model = (
        create_model(
            model_name
        )
    )


    fitted_warmup_model = (
        warmup_model.fit(
            base_train
        )
    )


    warmup_predictions = (
        fitted_warmup_model
        .transform(
            base_test
        )
    )


    warmup_prediction_count = (
        warmup_predictions.count()
    )


    print(
        "Warm-Up Predictions:",
        warmup_prediction_count
    )


    del warmup_predictions
    del fitted_warmup_model
    del warmup_model


    gc.collect()


print()

print(
    "ALGORITHM WARM-UP COMPLETE"
)


# ============================================================
# MAIN SCALABILITY EXPERIMENT
# ============================================================

for scale_factor in SCALE_FACTORS:

    print()

    print()

    print(
        "#" * 90
    )


    print(
        f"SCALABILITY LEVEL: "
        f"{scale_factor}x"
    )


    print(
        "#" * 90
    )


    # ========================================================
    # CREATE SCALED TRAINING DATA
    # ========================================================

    train_multiplier = (
        spark
        .range(
            scale_factor
        )
        .select(
            col(
                "id"
            ).alias(
                "_replication_id"
            )
        )
    )


    scaled_train = (
        base_train
        .crossJoin(
            train_multiplier
        )
        .drop(
            "_replication_id"
        )
        .repartition(
            NUM_PARTITIONS
        )
        .cache()
    )


    # ========================================================
    # CREATE SCALED TEST DATA
    # ========================================================

    test_multiplier = (
        spark
        .range(
            scale_factor
        )
        .select(
            col(
                "id"
            ).alias(
                "_replication_id"
            )
        )
    )


    scaled_test = (
        base_test
        .crossJoin(
            test_multiplier
        )
        .drop(
            "_replication_id"
        )
        .repartition(
            NUM_PARTITIONS
        )
        .cache()
    )


    # ========================================================
    # MATERIALIZE DATA
    # ========================================================

    scaled_train_rows = (
        scaled_train.count()
    )


    scaled_test_rows = (
        scaled_test.count()
    )


    scaled_total_rows = (
        scaled_train_rows
        +
        scaled_test_rows
    )


    print()


    print(
        "Synthetic Scale Factor:",
        scale_factor,
        "x"
    )


    print(
        "Training Rows:",
        scaled_train_rows
    )


    print(
        "Testing Rows:",
        scaled_test_rows
    )


    print(
        "Total Workload Rows:",
        scaled_total_rows
    )


    # ========================================================
    # MODELS
    # ========================================================

    for model_name in MODEL_NAMES:

        current_summary_key = (
            summary_key(
                scale_factor,
                model_name,
            )
        )


        # If this model/scale summary already exists,
        # the model has already fully completed.

        if (
            current_summary_key
            in
            completed_summary_keys
        ):

            print()

            print(
                f"SKIPPING COMPLETED MODEL: "
                f"{scale_factor}x - "
                f"{model_name}"
            )

            continue


        print()

        print("=" * 90)


        print(
            f"{scale_factor}x - "
            f"{model_name}"
        )


        print("=" * 90)


        # ====================================================
        # RUN OR SKIP INDIVIDUAL TRIALS
        # ====================================================

        for trial_number in range(
            1,
            NUM_TRIALS + 1,
        ):

            current_trial_key = (
                trial_key(
                    scale_factor,
                    model_name,
                    trial_number,
                )
            )


            if (
                current_trial_key
                in
                completed_trial_keys
            ):

                print()

                print(
                    f"Trial "
                    f"{trial_number}/"
                    f"{NUM_TRIALS} "
                    f"already completed - skipping."
                )

                continue


            print()


            print(
                f"Trial "
                f"{trial_number}/"
                f"{NUM_TRIALS}"
            )


            print(
                "-" * 90
            )


            # ================================================
            # CREATE FRESH MODEL
            # ================================================

            model = (
                create_model(
                    model_name
                )
            )


            # ================================================
            # TRAINING TIME
            # ================================================

            training_start = (
                time.perf_counter()
            )


            trained_model = (
                model.fit(
                    scaled_train
                )
            )


            training_end = (
                time.perf_counter()
            )


            training_time = (
                training_end
                -
                training_start
            )


            # ================================================
            # PREDICTION TIME
            # ================================================

            prediction_start = (
                time.perf_counter()
            )


            predictions = (
                trained_model
                .transform(
                    scaled_test
                )
                .cache()
            )


            prediction_rows = (
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


            # ================================================
            # THROUGHPUT
            # ================================================

            if training_time > 0:

                training_throughput = (
                    scaled_train_rows
                    /
                    training_time
                )

            else:

                training_throughput = 0.0


            if prediction_time > 0:

                prediction_throughput = (
                    prediction_rows
                    /
                    prediction_time
                )

            else:

                prediction_throughput = 0.0


            # ================================================
            # BASE TRIAL RESULT
            # ================================================

            trial_result = {
                "Scale_Factor":
                    scale_factor,

                "Model":
                    model_name,

                "Trial":
                    trial_number,

                "Base_Dataset_Rows":
                    base_total_rows,

                "Synthetic_Total_Rows":
                    scaled_total_rows,

                "Training_Rows":
                    scaled_train_rows,

                "Testing_Rows":
                    scaled_test_rows,

                "Feature_Count":
                    feature_count,

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

                "Training_Throughput_Rows_Per_Second":
                    round(
                        training_throughput,
                        2
                    ),

                "Prediction_Throughput_Rows_Per_Second":
                    round(
                        prediction_throughput,
                        2
                    ),

                "Accuracy":
                    "",

                "Precision_Churn":
                    "",

                "Recall_Churn":
                    "",

                "F1_Churn":
                    "",

                "Weighted_F1":
                    "",

                "ROC_AUC":
                    "",

                "True_Negatives":
                    "",

                "False_Positives":
                    "",

                "False_Negatives":
                    "",

                "True_Positives":
                    "",
            }


            # ================================================
            # PREDICTIVE METRICS FOR TRIAL 1
            # ================================================

            if trial_number == 1:

                accuracy = (
                    accuracy_evaluator
                    .evaluate(
                        predictions
                    )
                )


                weighted_f1 = (
                    weighted_f1_evaluator
                    .evaluate(
                        predictions
                    )
                )


                auc = (
                    auc_evaluator
                    .evaluate(
                        predictions
                    )
                )


                confusion_metrics = (
                    calculate_confusion_metrics(
                        predictions
                    )
                )


                trial_result[
                    "Accuracy"
                ] = round(
                    accuracy,
                    6
                )


                trial_result[
                    "Precision_Churn"
                ] = round(
                    confusion_metrics[
                        "Precision_Churn"
                    ],
                    6
                )


                trial_result[
                    "Recall_Churn"
                ] = round(
                    confusion_metrics[
                        "Recall_Churn"
                    ],
                    6
                )


                trial_result[
                    "F1_Churn"
                ] = round(
                    confusion_metrics[
                        "F1_Churn"
                    ],
                    6
                )


                trial_result[
                    "Weighted_F1"
                ] = round(
                    weighted_f1,
                    6
                )


                trial_result[
                    "ROC_AUC"
                ] = round(
                    auc,
                    6
                )


                trial_result[
                    "True_Negatives"
                ] = confusion_metrics[
                    "True_Negatives"
                ]


                trial_result[
                    "False_Positives"
                ] = confusion_metrics[
                    "False_Positives"
                ]


                trial_result[
                    "False_Negatives"
                ] = confusion_metrics[
                    "False_Negatives"
                ]


                trial_result[
                    "True_Positives"
                ] = confusion_metrics[
                    "True_Positives"
                ]


            # ================================================
            # DISPLAY TRIAL
            # ================================================

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
                "Training Throughput:",
                round(
                    training_throughput,
                    2
                ),
                "rows/second"
            )


            print(
                "Prediction Throughput:",
                round(
                    prediction_throughput,
                    2
                ),
                "rows/second"
            )


            # ================================================
            # SAVE CHECKPOINT IMMEDIATELY
            # ================================================

            trial_results.append(
                trial_result
            )


            completed_trial_keys.add(
                current_trial_key
            )


            save_csv(
                TRIAL_CHECKPOINT_FILE,
                trial_results,
                TRIAL_CHECKPOINT_FIELDS,
            )


            print(
                "Trial checkpoint saved."
            )


            # ================================================
            # CLEAN TRIAL
            # ================================================

            predictions.unpersist()


            del predictions
            del trained_model
            del model


            gc.collect()


        # ====================================================
        # REBUILD THIS MODEL'S TRIAL DATA
        # ====================================================

        current_model_trials = [
            row
            for row in trial_results
            if (
                int(
                    row[
                        "Scale_Factor"
                    ]
                )
                ==
                scale_factor
                and
                row[
                    "Model"
                ]
                ==
                model_name
            )
        ]


        current_model_trials.sort(
            key=lambda row:
                int(
                    row[
                        "Trial"
                    ]
                )
        )


        if len(
            current_model_trials
        ) != NUM_TRIALS:

            raise RuntimeError(
                f"Expected "
                f"{NUM_TRIALS} trials for "
                f"{scale_factor}x "
                f"{model_name}, but found "
                f"{len(current_model_trials)}."
            )


        # ====================================================
        # EXTRACT TIMING VALUES
        # ====================================================

        model_training_times = [
            float(
                row[
                    "Training_Time_Seconds"
                ]
            )
            for row in current_model_trials
        ]


        model_prediction_times = [
            float(
                row[
                    "Prediction_Time_Seconds"
                ]
            )
            for row in current_model_trials
        ]


        model_training_throughputs = [
            float(
                row[
                    "Training_Throughput_Rows_Per_Second"
                ]
            )
            for row in current_model_trials
        ]


        model_prediction_throughputs = [
            float(
                row[
                    "Prediction_Throughput_Rows_Per_Second"
                ]
            )
            for row in current_model_trials
        ]


        # ====================================================
        # GET METRICS FROM TRIAL 1
        # ====================================================

        trial_one = (
            current_model_trials[0]
        )


        if (
            trial_one[
                "Accuracy"
            ]
            ==
            ""
        ):

            raise RuntimeError(
                f"Predictive metrics are missing for "
                f"{scale_factor}x "
                f"{model_name} Trial 1."
            )


        model_metrics = {
            "Accuracy":
                float(
                    trial_one[
                        "Accuracy"
                    ]
                ),

            "Precision_Churn":
                float(
                    trial_one[
                        "Precision_Churn"
                    ]
                ),

            "Recall_Churn":
                float(
                    trial_one[
                        "Recall_Churn"
                    ]
                ),

            "F1_Churn":
                float(
                    trial_one[
                        "F1_Churn"
                    ]
                ),

            "Weighted_F1":
                float(
                    trial_one[
                        "Weighted_F1"
                    ]
                ),

            "ROC_AUC":
                float(
                    trial_one[
                        "ROC_AUC"
                    ]
                ),

            "True_Negatives":
                int(
                    float(
                        trial_one[
                            "True_Negatives"
                        ]
                    )
                ),

            "False_Positives":
                int(
                    float(
                        trial_one[
                            "False_Positives"
                        ]
                    )
                ),

            "False_Negatives":
                int(
                    float(
                        trial_one[
                            "False_Negatives"
                        ]
                    )
                ),

            "True_Positives":
                int(
                    float(
                        trial_one[
                            "True_Positives"
                        ]
                    )
                ),
        }


        # ====================================================
        # REPEATED TIMING STATISTICS
        # ====================================================

        mean_training_time = (
            calculate_mean(
                model_training_times
            )
        )


        std_training_time = (
            calculate_std(
                model_training_times
            )
        )


        mean_prediction_time = (
            calculate_mean(
                model_prediction_times
            )
        )


        std_prediction_time = (
            calculate_std(
                model_prediction_times
            )
        )


        mean_training_throughput = (
            calculate_mean(
                model_training_throughputs
            )
        )


        std_training_throughput = (
            calculate_std(
                model_training_throughputs
            )
        )


        mean_prediction_throughput = (
            calculate_mean(
                model_prediction_throughputs
            )
        )


        std_prediction_throughput = (
            calculate_std(
                model_prediction_throughputs
            )
        )


        # ====================================================
        # DISPLAY SUMMARY
        # ====================================================

        print()

        print(
            "-" * 90
        )


        print(
            f"{scale_factor}x - "
            f"{model_name} "
            f"REPEATED TRIAL SUMMARY"
        )


        print(
            "-" * 90
        )


        print(
            "\nPREDICTIVE PERFORMANCE"
        )


        print(
            "Accuracy:",
            round(
                model_metrics[
                    "Accuracy"
                ],
                4
            )
        )


        print(
            "Precision (Churn):",
            round(
                model_metrics[
                    "Precision_Churn"
                ],
                4
            )
        )


        print(
            "Recall (Churn):",
            round(
                model_metrics[
                    "Recall_Churn"
                ],
                4
            )
        )


        print(
            "F1 (Churn):",
            round(
                model_metrics[
                    "F1_Churn"
                ],
                4
            )
        )


        print(
            "Weighted F1:",
            round(
                model_metrics[
                    "Weighted_F1"
                ],
                4
            )
        )


        print(
            "ROC-AUC:",
            round(
                model_metrics[
                    "ROC_AUC"
                ],
                4
            )
        )


        print(
            "\nREPEATED COMPUTATIONAL PERFORMANCE"
        )


        print(
            "Mean Training Time:",
            round(
                mean_training_time,
                4
            ),
            "seconds"
        )


        print(
            "Training Time Standard Deviation:",
            round(
                std_training_time,
                4
            ),
            "seconds"
        )


        print(
            "Mean Prediction Time:",
            round(
                mean_prediction_time,
                4
            ),
            "seconds"
        )


        print(
            "Prediction Time Standard Deviation:",
            round(
                std_prediction_time,
                4
            ),
            "seconds"
        )


        print(
            "Mean Training Throughput:",
            round(
                mean_training_throughput,
                2
            ),
            "rows/second"
        )


        print(
            "Mean Prediction Throughput:",
            round(
                mean_prediction_throughput,
                2
            ),
            "rows/second"
        )


        # ====================================================
        # BUILD SUMMARY RESULT
        # ====================================================

        summary_result = {
            "Scale_Factor":
                scale_factor,

            "Model":
                model_name,

            "Trials":
                NUM_TRIALS,

            "Base_Dataset_Rows":
                base_total_rows,

            "Synthetic_Total_Rows":
                scaled_total_rows,

            "Training_Rows":
                scaled_train_rows,

            "Testing_Rows":
                scaled_test_rows,

            "Feature_Count":
                feature_count,

            "Accuracy":
                round(
                    model_metrics[
                        "Accuracy"
                    ],
                    6
                ),

            "Precision_Churn":
                round(
                    model_metrics[
                        "Precision_Churn"
                    ],
                    6
                ),

            "Recall_Churn":
                round(
                    model_metrics[
                        "Recall_Churn"
                    ],
                    6
                ),

            "F1_Churn":
                round(
                    model_metrics[
                        "F1_Churn"
                    ],
                    6
                ),

            "Weighted_F1":
                round(
                    model_metrics[
                        "Weighted_F1"
                    ],
                    6
                ),

            "ROC_AUC":
                round(
                    model_metrics[
                        "ROC_AUC"
                    ],
                    6
                ),

            "True_Negatives":
                model_metrics[
                    "True_Negatives"
                ],

            "False_Positives":
                model_metrics[
                    "False_Positives"
                ],

            "False_Negatives":
                model_metrics[
                    "False_Negatives"
                ],

            "True_Positives":
                model_metrics[
                    "True_Positives"
                ],

            "Training_Time_Run_1":
                round(
                    model_training_times[0],
                    6
                ),

            "Training_Time_Run_2":
                round(
                    model_training_times[1],
                    6
                ),

            "Training_Time_Run_3":
                round(
                    model_training_times[2],
                    6
                ),

            "Mean_Training_Time_Seconds":
                round(
                    mean_training_time,
                    6
                ),

            "Std_Training_Time_Seconds":
                round(
                    std_training_time,
                    6
                ),

            "Prediction_Time_Run_1":
                round(
                    model_prediction_times[0],
                    6
                ),

            "Prediction_Time_Run_2":
                round(
                    model_prediction_times[1],
                    6
                ),

            "Prediction_Time_Run_3":
                round(
                    model_prediction_times[2],
                    6
                ),

            "Mean_Prediction_Time_Seconds":
                round(
                    mean_prediction_time,
                    6
                ),

            "Std_Prediction_Time_Seconds":
                round(
                    std_prediction_time,
                    6
                ),

            "Mean_Training_Throughput_Rows_Per_Second":
                round(
                    mean_training_throughput,
                    2
                ),

            "Std_Training_Throughput_Rows_Per_Second":
                round(
                    std_training_throughput,
                    2
                ),

            "Mean_Prediction_Throughput_Rows_Per_Second":
                round(
                    mean_prediction_throughput,
                    2
                ),

            "Std_Prediction_Throughput_Rows_Per_Second":
                round(
                    std_prediction_throughput,
                    2
                ),
        }


        summary_results.append(
            summary_result
        )


        completed_summary_keys.add(
            current_summary_key
        )


        # Save summary checkpoint immediately.

        save_csv(
            SUMMARY_CHECKPOINT_FILE,
            summary_results,
            SUMMARY_FIELDS,
        )


        print(
            "Model/scale summary checkpoint saved."
        )


    # ========================================================
    # CLEAN CURRENT SCALE
    # ========================================================

    scaled_train.unpersist()

    scaled_test.unpersist()


    del scaled_train
    del scaled_test

    del train_multiplier
    del test_multiplier


    gc.collect()


    print()


    print(
        f"{scale_factor}x SCALE COMPLETED"
    )


# ============================================================
# VERIFY COMPLETE EXPERIMENT
# ============================================================

expected_trials = (
    len(
        SCALE_FACTORS
    )
    *
    len(
        MODEL_NAMES
    )
    *
    NUM_TRIALS
)


expected_summaries = (
    len(
        SCALE_FACTORS
    )
    *
    len(
        MODEL_NAMES
    )
)


print()

print("=" * 90)

print(
    "VERIFYING EXPERIMENT COMPLETION"
)

print("=" * 90)


print(
    "Expected Trial Rows:",
    expected_trials
)


print(
    "Actual Trial Rows:",
    len(
        trial_results
    )
)


print(
    "Expected Summary Rows:",
    expected_summaries
)


print(
    "Actual Summary Rows:",
    len(
        summary_results
    )
)


if len(
    trial_results
) != expected_trials:

    raise RuntimeError(
        "Experiment is not complete. "
        "Trial checkpoint has been preserved."
    )


if len(
    summary_results
) != expected_summaries:

    raise RuntimeError(
        "Experiment is not complete. "
        "Summary checkpoint has been preserved."
    )


print(
    "EXPERIMENT COMPLETION VERIFIED"
)


# ============================================================
# SORT FINAL RESULTS
# ============================================================

scale_order = {
    scale:
        index
    for index, scale in enumerate(
        SCALE_FACTORS
    )
}


model_order = {
    model:
        index
    for index, model in enumerate(
        MODEL_NAMES
    )
}


trial_results.sort(
    key=lambda row: (
        scale_order[
            int(
                row[
                    "Scale_Factor"
                ]
            )
        ],
        model_order[
            row[
                "Model"
            ]
        ],
        int(
            row[
                "Trial"
            ]
        ),
    )
)


summary_results.sort(
    key=lambda row: (
        scale_order[
            int(
                row[
                    "Scale_Factor"
                ]
            )
        ],
        model_order[
            row[
                "Model"
            ]
        ],
    )
)


# ============================================================
# SAVE FINAL RAW TRIAL RESULTS
# ============================================================

print()

print("=" * 90)

print(
    "SAVING FINAL RAW TRIAL RESULTS"
)

print("=" * 90)


save_csv(
    TRIAL_RESULTS_FILE,
    trial_results,
    TRIAL_FINAL_FIELDS,
)


print(
    "RAW TRIAL RESULTS SAVED SUCCESSFULLY"
)


print(
    TRIAL_RESULTS_FILE
)


# ============================================================
# SAVE FINAL SUMMARY RESULTS
# ============================================================

print()

print("=" * 90)

print(
    "SAVING FINAL REPEATED SUMMARY RESULTS"
)

print("=" * 90)


save_csv(
    SUMMARY_RESULTS_FILE,
    summary_results,
    SUMMARY_FIELDS,
)


print(
    "REPEATED SUMMARY RESULTS SAVED SUCCESSFULLY"
)


print(
    SUMMARY_RESULTS_FILE
)


# ============================================================
# FINAL SUMMARY TABLE
# ============================================================

print()

print("=" * 160)

print(
    "FINAL REPEATED SCALABILITY SUMMARY"
)

print("=" * 160)


print(
    f"{'Scale':<8}"
    f"{'Model':<25}"
    f"{'Rows':<12}"
    f"{'Accuracy':<11}"
    f"{'AUC':<10}"
    f"{'Mean Train':<14}"
    f"{'Train SD':<12}"
    f"{'Mean Pred':<14}"
    f"{'Pred SD':<12}"
    f"{'Train r/s':<14}"
    f"{'Pred r/s':<14}"
)


print(
    "-" * 160
)


for result in summary_results:

    print(
        f"{str(result['Scale_Factor']) + 'x':<8}"
        f"{result['Model']:<25}"
        f"{result['Synthetic_Total_Rows']:<12}"
        f"{result['Accuracy']:<11.4f}"
        f"{result['ROC_AUC']:<10.4f}"
        f"{result['Mean_Training_Time_Seconds']:<14.4f}"
        f"{result['Std_Training_Time_Seconds']:<12.4f}"
        f"{result['Mean_Prediction_Time_Seconds']:<14.4f}"
        f"{result['Std_Prediction_Time_Seconds']:<12.4f}"
        f"{result['Mean_Training_Throughput_Rows_Per_Second']:<14.2f}"
        f"{result['Mean_Prediction_Throughput_Rows_Per_Second']:<14.2f}"
    )


print(
    "=" * 160
)


# ============================================================
# RESEARCH NOTES
# ============================================================

print()

print("=" * 90)

print(
    "IMPORTANT RESEARCH NOTES"
)

print("=" * 90)


print(
    "1. Larger workloads were created by replicating "
    "the original observations."
)


print()


print(
    "2. Replicated records are synthetic computational workloads."
)


print()


print(
    "3. They must NOT be described as new independent "
    "customer observations."
)


print()


print(
    "4. Predictive generalization must be based on the "
    "original held-out test experiment."
)


print()


print(
    "5. Scalability conclusions should use the repeated "
    "timing means and standard deviations."
)


print()


print(
    "6. This experiment used Spark local[2], so it measures "
    "local parallel scalability,"
)


print(
    "   not multi-node cluster scalability."
)


# ============================================================
# REMOVE CHECKPOINT FILES AFTER SUCCESS
# ============================================================

print()

print("=" * 90)

print(
    "CLEANING CHECKPOINT FILES"
)

print("=" * 90)


if TRIAL_CHECKPOINT_FILE.exists():

    TRIAL_CHECKPOINT_FILE.unlink()

    print(
        "Removed:",
        TRIAL_CHECKPOINT_FILE
    )


if SUMMARY_CHECKPOINT_FILE.exists():

    SUMMARY_CHECKPOINT_FILE.unlink()

    print(
        "Removed:",
        SUMMARY_CHECKPOINT_FILE
    )


# ============================================================
# CLEANUP
# ============================================================

base_train.unpersist()

base_test.unpersist()


spark.stop()


print()

print("=" * 90)

print(
    "REPEATED SCALABILITY EXPERIMENT COMPLETE"
)

print("=" * 90)


print(
    "Raw Trial Results:"
)

print(
    TRIAL_RESULTS_FILE
)


print()


print(
    "Summary Results:"
)

print(
    SUMMARY_RESULTS_FILE
)


print()


print(
    "SPARK STOPPED"
)


print("=" * 90)