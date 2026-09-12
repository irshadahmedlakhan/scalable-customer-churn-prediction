import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_PATH = Path(__file__).resolve().parents[1]

RESULTS_PATH = (
    PROJECT_PATH
    / "results"
)

MODEL_COMPARISON_FILE = (
    RESULTS_PATH
    / "model_comparison.csv"
)

SCALABILITY_RESULTS_FILE = (
    RESULTS_PATH
    / "scalability_repeated_results.csv"
)

FIGURES_PATH = (
    RESULTS_PATH
    / "figures"
)


FIGURES_PATH.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# EXPECTED MODELS
# ============================================================

MODEL_DISPLAY_NAMES = {
    "Logistic_Regression":
        "Logistic Regression",

    "Decision_Tree":
        "Decision Tree",

    "Random_Forest":
        "Random Forest",

    "Gradient_Boosted_Tree":
        "Gradient Boosted Tree",
}


MODEL_ORDER = [
    "Logistic_Regression",
    "Decision_Tree",
    "Random_Forest",
    "Gradient_Boosted_Tree",
]


EXPECTED_SCALE_FACTORS = [
    1,
    5,
    10,
    20,
    50,
    100,
]


# ============================================================
# START MESSAGE
# ============================================================

print()

print("=" * 90)

print(
    "CUSTOMER CHURN RESEARCH - RESULTS VISUALIZATION"
)

print("=" * 90)


print(
    "\nModel Comparison File:"
)

print(
    MODEL_COMPARISON_FILE
)


print(
    "\nScalability Results File:"
)

print(
    SCALABILITY_RESULTS_FILE
)


print(
    "\nFigures Output Directory:"
)

print(
    FIGURES_PATH
)


print("=" * 90)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def read_csv_file(
    file_path
):

    if not file_path.exists():

        raise FileNotFoundError(
            f"Required file not found:\n"
            f"{file_path}"
        )


    with open(
        file_path,
        "r",
        encoding="utf-8",
        newline="",
    ) as csv_file:

        reader = csv.DictReader(
            csv_file
        )

        rows = list(
            reader
        )


    if not rows:

        raise ValueError(
            f"CSV file is empty:\n"
            f"{file_path}"
        )


    return rows


def get_float(
    row,
    key
):

    if key not in row:

        raise KeyError(
            f"Column '{key}' not found in CSV."
        )


    value = row[
        key
    ]


    if value is None or value == "":

        raise ValueError(
            f"Column '{key}' contains an empty value."
        )


    return float(
        value
    )


def get_int(
    row,
    key
):

    return int(
        get_float(
            row,
            key
        )
    )


def find_column(
    rows,
    candidates,
    description
):

    available_columns = (
        rows[0].keys()
    )


    for candidate in candidates:

        if candidate in available_columns:

            return candidate


    raise KeyError(
        f"Could not find column for {description}.\n"
        f"Tried: {candidates}\n"
        f"Available columns: "
        f"{list(available_columns)}"
    )


def save_figure(
    file_name
):

    output_path = (
        FIGURES_PATH
        / file_name
    )


    plt.tight_layout()


    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )


    plt.close()


    print(
        "Saved:",
        output_path
    )


def add_bar_value_labels(
    bars,
    values,
    decimals=4,
    suffix="",
):

    maximum_value = max(
        values
    )


    if maximum_value > 0:

        offset = (
            maximum_value
            *
            0.02
        )

    else:

        offset = 0.01


    for bar, value in zip(
        bars,
        values
    ):

        plt.text(
            bar.get_x()
            +
            bar.get_width()
            /
            2,

            value
            +
            offset,

            f"{value:.{decimals}f}"
            f"{suffix}",

            ha="center",
            va="bottom",
        )


# ============================================================
# LOAD BASELINE MODEL RESULTS
# ============================================================

print()

print("=" * 90)

print(
    "LOADING MODEL COMPARISON RESULTS"
)

print("=" * 90)


model_rows = (
    read_csv_file(
        MODEL_COMPARISON_FILE
    )
)


print(
    "Rows Loaded:",
    len(
        model_rows
    )
)


print(
    "Columns:"
)


print(
    list(
        model_rows[0].keys()
    )
)


# ============================================================
# LOAD SCALABILITY RESULTS
# ============================================================

print()

print("=" * 90)

print(
    "LOADING REPEATED SCALABILITY RESULTS"
)

print("=" * 90)


scalability_rows = (
    read_csv_file(
        SCALABILITY_RESULTS_FILE
    )
)


print(
    "Rows Loaded:",
    len(
        scalability_rows
    )
)


print(
    "Columns:"
)


print(
    list(
        scalability_rows[0].keys()
    )
)


# ============================================================
# VALIDATE SCALABILITY RESULT COUNT
# ============================================================

expected_summary_rows = (
    len(
        EXPECTED_SCALE_FACTORS
    )
    *
    len(
        MODEL_ORDER
    )
)


if len(
    scalability_rows
) != expected_summary_rows:

    print()

    print(
        "WARNING:"
    )

    print(
        f"Expected "
        f"{expected_summary_rows} scalability summary rows "
        f"but found "
        f"{len(scalability_rows)}."
    )


else:

    print(
        "Scalability summary row count verified:",
        expected_summary_rows
    )


# ============================================================
# DETECT MODEL COLUMN
# ============================================================

model_column_candidates = [
    "Model",
    "model",
    "Algorithm",
    "algorithm",
]


model_column = None


for candidate in model_column_candidates:

    if candidate in model_rows[0]:

        model_column = candidate

        break


if model_column is None:

    raise KeyError(
        "Could not find the model name column "
        "in model_comparison.csv"
    )


# ============================================================
# ORGANIZE BASELINE MODEL RESULTS
# ============================================================

baseline_by_model = {}


for row in model_rows:

    model_name = (
        row[
            model_column
        ]
    )


    baseline_by_model[
        model_name
    ] = row


available_models = [
    model
    for model in MODEL_ORDER
    if model in baseline_by_model
]


if not available_models:

    raise ValueError(
        "No expected models were found "
        "in model_comparison.csv"
    )


if len(
    available_models
) != len(
    MODEL_ORDER
):

    print()

    print(
        "WARNING:"
    )

    print(
        "Not all expected baseline models were found."
    )

    print(
        "Available:",
        available_models
    )


display_names = [
    MODEL_DISPLAY_NAMES[
        model
    ]
    for model in available_models
]


# ============================================================
# FIND BASELINE METRIC COLUMNS
# ============================================================

accuracy_column = find_column(
    model_rows,
    [
        "Accuracy",
        "accuracy",
    ],
    "Accuracy",
)


auc_column = find_column(
    model_rows,
    [
        "ROC_AUC",
        "AUC",
        "ROC-AUC",
        "roc_auc",
    ],
    "ROC-AUC",
)


f1_churn_column = find_column(
    model_rows,
    [
        "F1_Churn",
        "Churn_F1",
        "F1 (Churn)",
        "F1",
    ],
    "Churn F1",
)


training_time_column = find_column(
    model_rows,
    [
        "Training_Time_Seconds",
        "Training_Time",
        "Train_Time",
        "Training Time",
    ],
    "Training Time",
)


prediction_time_column = find_column(
    model_rows,
    [
        "Prediction_Time_Seconds",
        "Prediction_Time",
        "Predict_Time",
        "Prediction Time",
    ],
    "Prediction Time",
)


# ============================================================
# EXTRACT BASELINE VALUES
# ============================================================

baseline_accuracy = [
    float(
        baseline_by_model[
            model
        ][
            accuracy_column
        ]
    )
    for model in available_models
]


baseline_auc = [
    float(
        baseline_by_model[
            model
        ][
            auc_column
        ]
    )
    for model in available_models
]


baseline_f1 = [
    float(
        baseline_by_model[
            model
        ][
            f1_churn_column
        ]
    )
    for model in available_models
]


baseline_training_time = [
    float(
        baseline_by_model[
            model
        ][
            training_time_column
        ]
    )
    for model in available_models
]


baseline_prediction_time = [
    float(
        baseline_by_model[
            model
        ][
            prediction_time_column
        ]
    )
    for model in available_models
]


# ============================================================
# ORGANIZE SCALABILITY RESULTS
# ============================================================

scalability_by_model = {
    model:
        []
    for model in MODEL_ORDER
}


for row in scalability_rows:

    model_name = row.get(
        "Model"
    )


    if model_name in scalability_by_model:

        scalability_by_model[
            model_name
        ].append(
            row
        )


for model_name in MODEL_ORDER:

    scalability_by_model[
        model_name
    ].sort(
        key=lambda row:
            get_int(
                row,
                "Scale_Factor"
            )
    )


# ============================================================
# VALIDATE SCALE FACTORS FOR EACH MODEL
# ============================================================

for model_name in MODEL_ORDER:

    rows = (
        scalability_by_model[
            model_name
        ]
    )


    scales = [
        get_int(
            row,
            "Scale_Factor"
        )
        for row in rows
    ]


    if scales != EXPECTED_SCALE_FACTORS:

        print()

        print(
            "WARNING:"
        )

        print(
            MODEL_DISPLAY_NAMES[
                model_name
            ],
            "has scale factors:",
            scales
        )


# ============================================================
# GENERIC SCALABILITY PLOT
# ============================================================

def create_scalability_plot(
    y_column,
    title,
    y_label,
    file_name,
):

    plt.figure(
        figsize=(
            11,
            7,
        )
    )


    for model_name in MODEL_ORDER:

        rows = (
            scalability_by_model[
                model_name
            ]
        )


        if not rows:

            continue


        workload_rows = [
            get_int(
                row,
                "Synthetic_Total_Rows"
            )
            for row in rows
        ]


        values = [
            get_float(
                row,
                y_column
            )
            for row in rows
        ]


        plt.plot(
            workload_rows,
            values,
            marker="o",
            linewidth=2,
            label=
                MODEL_DISPLAY_NAMES[
                    model_name
                ],
        )


    plt.title(
        title
    )


    plt.xlabel(
        "Synthetic Workload Size (Rows)"
    )


    plt.ylabel(
        y_label
    )


    plt.grid(
        alpha=0.25
    )


    plt.legend()


    plt.ticklabel_format(
        style="plain",
        axis="x",
    )


    save_figure(
        file_name
    )


# ============================================================
# FIGURE 1
# BASELINE PREDICTIVE METRICS COMPARISON
# ============================================================

print()

print("=" * 90)

print(
    "CREATING FIGURE 1 - BASELINE PREDICTIVE METRICS"
)

print("=" * 90)


x_positions = np.arange(
    len(
        display_names
    )
)


bar_width = 0.25


plt.figure(
    figsize=(
        12,
        7,
    )
)


plt.bar(
    x_positions
    -
    bar_width,
    baseline_accuracy,
    width=bar_width,
    label="Accuracy",
)


plt.bar(
    x_positions,
    baseline_f1,
    width=bar_width,
    label="Churn F1",
)


plt.bar(
    x_positions
    +
    bar_width,
    baseline_auc,
    width=bar_width,
    label="ROC-AUC",
)


plt.title(
    "Baseline Predictive Performance Comparison"
)


plt.xlabel(
    "Machine Learning Model"
)


plt.ylabel(
    "Score"
)


plt.ylim(
    0,
    1,
)


plt.xticks(
    x_positions,
    display_names,
    rotation=15,
)


plt.grid(
    axis="y",
    alpha=0.25,
)


plt.legend()


save_figure(
    "baseline_predictive_metrics_comparison.png"
)


# ============================================================
# FIGURE 2
# BASELINE ACCURACY
# ============================================================

print()

print("=" * 90)

print(
    "CREATING FIGURE 2 - MODEL ACCURACY"
)

print("=" * 90)


plt.figure(
    figsize=(
        10,
        6,
    )
)


bars = plt.bar(
    display_names,
    baseline_accuracy,
)


plt.title(
    "Baseline Model Accuracy Comparison"
)


plt.xlabel(
    "Machine Learning Model"
)


plt.ylabel(
    "Accuracy"
)


plt.ylim(
    0,
    1,
)


plt.xticks(
    rotation=15
)


plt.grid(
    axis="y",
    alpha=0.25,
)


add_bar_value_labels(
    bars,
    baseline_accuracy,
    decimals=4,
)


save_figure(
    "model_accuracy_comparison.png"
)


# ============================================================
# FIGURE 3
# BASELINE ROC-AUC
# ============================================================

print()

print("=" * 90)

print(
    "CREATING FIGURE 3 - MODEL ROC-AUC"
)

print("=" * 90)


plt.figure(
    figsize=(
        10,
        6,
    )
)


bars = plt.bar(
    display_names,
    baseline_auc,
)


plt.title(
    "Baseline Model ROC-AUC Comparison"
)


plt.xlabel(
    "Machine Learning Model"
)


plt.ylabel(
    "ROC-AUC"
)


plt.ylim(
    0,
    1,
)


plt.xticks(
    rotation=15
)


plt.grid(
    axis="y",
    alpha=0.25,
)


add_bar_value_labels(
    bars,
    baseline_auc,
    decimals=4,
)


save_figure(
    "model_auc_comparison.png"
)


# ============================================================
# FIGURE 4
# BASELINE CHURN F1
# ============================================================

print()

print("=" * 90)

print(
    "CREATING FIGURE 4 - CHURN F1"
)

print("=" * 90)


plt.figure(
    figsize=(
        10,
        6,
    )
)


bars = plt.bar(
    display_names,
    baseline_f1,
)


plt.title(
    "Baseline Churn-Class F1 Score Comparison"
)


plt.xlabel(
    "Machine Learning Model"
)


plt.ylabel(
    "Churn F1 Score"
)


plt.ylim(
    0,
    1,
)


plt.xticks(
    rotation=15
)


plt.grid(
    axis="y",
    alpha=0.25,
)


add_bar_value_labels(
    bars,
    baseline_f1,
    decimals=4,
)


save_figure(
    "churn_f1_comparison.png"
)


# ============================================================
# FIGURE 5
# BASELINE TRAINING TIME
# ============================================================

print()

print("=" * 90)

print(
    "CREATING FIGURE 5 - BASELINE TRAINING TIME"
)

print("=" * 90)


plt.figure(
    figsize=(
        10,
        6,
    )
)


bars = plt.bar(
    display_names,
    baseline_training_time,
)


plt.title(
    "Baseline Model Training Time"
)


plt.xlabel(
    "Machine Learning Model"
)


plt.ylabel(
    "Training Time (Seconds)"
)


plt.xticks(
    rotation=15
)


plt.grid(
    axis="y",
    alpha=0.25,
)


add_bar_value_labels(
    bars,
    baseline_training_time,
    decimals=2,
    suffix="s",
)


save_figure(
    "baseline_training_time_comparison.png"
)


# ============================================================
# FIGURE 6
# BASELINE PREDICTION TIME
# ============================================================

print()

print("=" * 90)

print(
    "CREATING FIGURE 6 - BASELINE PREDICTION TIME"
)

print("=" * 90)


plt.figure(
    figsize=(
        10,
        6,
    )
)


bars = plt.bar(
    display_names,
    baseline_prediction_time,
)


plt.title(
    "Baseline Model Prediction Time"
)


plt.xlabel(
    "Machine Learning Model"
)


plt.ylabel(
    "Prediction Time (Seconds)"
)


plt.xticks(
    rotation=15
)


plt.grid(
    axis="y",
    alpha=0.25,
)


add_bar_value_labels(
    bars,
    baseline_prediction_time,
    decimals=3,
    suffix="s",
)


save_figure(
    "baseline_prediction_time_comparison.png"
)


# ============================================================
# FIGURE 7
# TRAINING TIME SCALABILITY
# ============================================================

print()

print("=" * 90)

print(
    "CREATING FIGURE 7 - TRAINING TIME SCALABILITY"
)

print("=" * 90)


create_scalability_plot(
    y_column=
        "Mean_Training_Time_Seconds",

    title=
        "Training Time vs Synthetic Workload Size",

    y_label=
        "Mean Training Time (Seconds)",

    file_name=
        "training_time_scalability.png",
)


# ============================================================
# FIGURE 8
# TRAINING TIME WITH ERROR BARS
# ============================================================

print()

print("=" * 90)

print(
    "CREATING FIGURE 8 - TRAINING TIME ERROR BARS"
)

print("=" * 90)


plt.figure(
    figsize=(
        11,
        7,
    )
)


for model_name in MODEL_ORDER:

    rows = (
        scalability_by_model[
            model_name
        ]
    )


    if not rows:

        continue


    workload_rows = [
        get_int(
            row,
            "Synthetic_Total_Rows"
        )
        for row in rows
    ]


    mean_training_times = [
        get_float(
            row,
            "Mean_Training_Time_Seconds"
        )
        for row in rows
    ]


    training_std = [
        get_float(
            row,
            "Std_Training_Time_Seconds"
        )
        for row in rows
    ]


    plt.errorbar(
        workload_rows,
        mean_training_times,
        yerr=training_std,
        marker="o",
        linewidth=2,
        capsize=5,
        label=
            MODEL_DISPLAY_NAMES[
                model_name
            ],
    )


plt.title(
    "Mean Training Time with Standard Deviation"
)


plt.xlabel(
    "Synthetic Workload Size (Rows)"
)


plt.ylabel(
    "Training Time (Seconds)"
)


plt.grid(
    alpha=0.25
)


plt.legend()


plt.ticklabel_format(
    style="plain",
    axis="x",
)


save_figure(
    "training_time_with_error_bars.png"
)


# ============================================================
# FIGURE 9
# PREDICTION TIME SCALABILITY
# ============================================================

print()

print("=" * 90)

print(
    "CREATING FIGURE 9 - PREDICTION TIME SCALABILITY"
)

print("=" * 90)


create_scalability_plot(
    y_column=
        "Mean_Prediction_Time_Seconds",

    title=
        "Prediction Time vs Synthetic Workload Size",

    y_label=
        "Mean Prediction Time (Seconds)",

    file_name=
        "prediction_time_scalability.png",
)


# ============================================================
# FIGURE 10
# PREDICTION TIME WITH ERROR BARS
# ============================================================

print()

print("=" * 90)

print(
    "CREATING FIGURE 10 - PREDICTION TIME ERROR BARS"
)

print("=" * 90)


plt.figure(
    figsize=(
        11,
        7,
    )
)


for model_name in MODEL_ORDER:

    rows = (
        scalability_by_model[
            model_name
        ]
    )


    if not rows:

        continue


    workload_rows = [
        get_int(
            row,
            "Synthetic_Total_Rows"
        )
        for row in rows
    ]


    mean_prediction_times = [
        get_float(
            row,
            "Mean_Prediction_Time_Seconds"
        )
        for row in rows
    ]


    prediction_std = [
        get_float(
            row,
            "Std_Prediction_Time_Seconds"
        )
        for row in rows
    ]


    plt.errorbar(
        workload_rows,
        mean_prediction_times,
        yerr=prediction_std,
        marker="o",
        linewidth=2,
        capsize=5,
        label=
            MODEL_DISPLAY_NAMES[
                model_name
            ],
    )


plt.title(
    "Mean Prediction Time with Standard Deviation"
)


plt.xlabel(
    "Synthetic Workload Size (Rows)"
)


plt.ylabel(
    "Prediction Time (Seconds)"
)


plt.grid(
    alpha=0.25
)


plt.legend()


plt.ticklabel_format(
    style="plain",
    axis="x",
)


save_figure(
    "prediction_time_with_error_bars.png"
)


# ============================================================
# FIGURE 11
# TRAINING THROUGHPUT
# ============================================================

print()

print("=" * 90)

print(
    "CREATING FIGURE 11 - TRAINING THROUGHPUT"
)

print("=" * 90)


create_scalability_plot(
    y_column=
        "Mean_Training_Throughput_Rows_Per_Second",

    title=
        "Training Throughput vs Synthetic Workload Size",

    y_label=
        "Mean Training Throughput (Rows/Second)",

    file_name=
        "training_throughput_scalability.png",
)


# ============================================================
# FIGURE 12
# PREDICTION THROUGHPUT
# ============================================================

print()

print("=" * 90)

print(
    "CREATING FIGURE 12 - PREDICTION THROUGHPUT"
)

print("=" * 90)


create_scalability_plot(
    y_column=
        "Mean_Prediction_Throughput_Rows_Per_Second",

    title=
        "Prediction Throughput vs Synthetic Workload Size",

    y_label=
        "Mean Prediction Throughput (Rows/Second)",

    file_name=
        "prediction_throughput_scalability.png",
)


# ============================================================
# FIND LARGEST SCALE RESULTS
# ============================================================

largest_scale_rows = []


for model_name in MODEL_ORDER:

    rows = (
        scalability_by_model[
            model_name
        ]
    )


    if not rows:

        continue


    largest_row = max(
        rows,
        key=lambda row:
            get_int(
                row,
                "Scale_Factor"
            ),
    )


    largest_scale_rows.append(
        largest_row
    )


if not largest_scale_rows:

    raise ValueError(
        "Could not find largest-scale scalability results."
    )


largest_scale = max(
    get_int(
        row,
        "Scale_Factor"
    )
    for row in largest_scale_rows
)


largest_models = [
    MODEL_DISPLAY_NAMES[
        row[
            "Model"
        ]
    ]
    for row in largest_scale_rows
]


# ============================================================
# FIGURE 13
# LARGEST-SCALE TRAINING TIME
# ============================================================

print()

print("=" * 90)

print(
    f"CREATING FIGURE 13 - "
    f"{largest_scale}x TRAINING TIME"
)

print("=" * 90)


largest_training_times = [
    get_float(
        row,
        "Mean_Training_Time_Seconds"
    )
    for row in largest_scale_rows
]


largest_training_std = [
    get_float(
        row,
        "Std_Training_Time_Seconds"
    )
    for row in largest_scale_rows
]


plt.figure(
    figsize=(
        10,
        6,
    )
)


bars = plt.bar(
    largest_models,
    largest_training_times,
    yerr=
        largest_training_std,
    capsize=6,
)


plt.title(
    f"Training Time at "
    f"{largest_scale}x Synthetic Workload"
)


plt.xlabel(
    "Machine Learning Model"
)


plt.ylabel(
    "Mean Training Time (Seconds)"
)


plt.xticks(
    rotation=15
)


plt.grid(
    axis="y",
    alpha=0.25,
)


add_bar_value_labels(
    bars,
    largest_training_times,
    decimals=2,
    suffix="s",
)


save_figure(
    "100x_training_time_comparison.png"
)


# ============================================================
# FIGURE 14
# LARGEST-SCALE PREDICTION TIME
# ============================================================

print()

print("=" * 90)

print(
    f"CREATING FIGURE 14 - "
    f"{largest_scale}x PREDICTION TIME"
)

print("=" * 90)


largest_prediction_times = [
    get_float(
        row,
        "Mean_Prediction_Time_Seconds"
    )
    for row in largest_scale_rows
]


largest_prediction_std = [
    get_float(
        row,
        "Std_Prediction_Time_Seconds"
    )
    for row in largest_scale_rows
]


plt.figure(
    figsize=(
        10,
        6,
    )
)


bars = plt.bar(
    largest_models,
    largest_prediction_times,
    yerr=
        largest_prediction_std,
    capsize=6,
)


plt.title(
    f"Prediction Time at "
    f"{largest_scale}x Synthetic Workload"
)


plt.xlabel(
    "Machine Learning Model"
)


plt.ylabel(
    "Mean Prediction Time (Seconds)"
)


plt.xticks(
    rotation=15
)


plt.grid(
    axis="y",
    alpha=0.25,
)


add_bar_value_labels(
    bars,
    largest_prediction_times,
    decimals=2,
    suffix="s",
)


save_figure(
    "100x_prediction_time_comparison.png"
)


# ============================================================
# FINAL SUMMARY
# ============================================================

generated_figures = [
    "baseline_predictive_metrics_comparison.png",
    "model_accuracy_comparison.png",
    "model_auc_comparison.png",
    "churn_f1_comparison.png",
    "baseline_training_time_comparison.png",
    "baseline_prediction_time_comparison.png",
    "training_time_scalability.png",
    "training_time_with_error_bars.png",
    "prediction_time_scalability.png",
    "prediction_time_with_error_bars.png",
    "training_throughput_scalability.png",
    "prediction_throughput_scalability.png",
    "100x_training_time_comparison.png",
    "100x_prediction_time_comparison.png",
]


print()

print("=" * 90)

print(
    "VISUALIZATION COMPLETE"
)

print("=" * 90)


print(
    "\nFigures created in:"
)


print(
    FIGURES_PATH
)


print(
    "\nGenerated Figures:"
)


for figure_name in generated_figures:

    figure_path = (
        FIGURES_PATH
        / figure_name
    )


    if figure_path.exists():

        print(
            "✓",
            figure_name
        )

    else:

        print(
            "✗ MISSING:",
            figure_name
        )


print()

print(
    "Total Expected Figures:",
    len(
        generated_figures
    )
)


existing_figure_count = sum(
    1
    for figure_name in generated_figures
    if (
        FIGURES_PATH
        /
        figure_name
    ).exists()
)


print(
    "Total Figures Verified:",
    existing_figure_count
)


if existing_figure_count != len(
    generated_figures
):

    raise RuntimeError(
        "One or more expected figures were not generated."
    )


print()

print("=" * 90)

print(
    "ALL 14 FIGURES GENERATED SUCCESSFULLY"
)

print("=" * 90)