# Scalable Customer Churn Prediction Using Big Data and Machine Learning

## Overview

This project investigates customer churn prediction using Apache Spark and machine learning.

The main objective is to compare multiple machine-learning algorithms in terms of both:

- Predictive performance
- Computational scalability

Four classification algorithms are implemented using Apache Spark MLlib:

- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosted Tree

The original Telco Customer Churn dataset contained 7,043 customer records. After data cleaning, 7,032 observations were used for modelling.

In addition to evaluating predictive performance on a held-out test set, the project investigates how model training and prediction times change as computational workload size increases.

Synthetic workloads ranging from 1x to 100x were created by replicating the original training and testing observations separately.

The largest computational workload contains 703,200 rows.

> **Important:** The scaled workloads were created through replication for computational scalability testing. They do not represent 703,200 independent customers.

---

## Research Questions

This project investigates the following research questions:

**RQ1:** How accurately can machine-learning algorithms predict customer churn?

**RQ2:** How do Logistic Regression, Decision Tree, Random Forest, and Gradient Boosted Tree compare in predictive performance?

**RQ3:** How does increasing dataset workload size affect training and prediction time using Apache Spark?

**RQ4:** Which model provides the best balance between predictive performance and computational scalability?

---

## Technologies

The project uses:

- Python
- PySpark
- Apache Spark
- Spark MLlib
- Spark SQL
- Parquet
- NumPy
- Matplotlib

Spark was executed using:

```text
local[2]
```

This configuration provides two local execution threads.

Therefore, the scalability results represent local Spark parallel workload behaviour on a single machine rather than multi-node cluster scalability.

---

## Dataset

The project uses the Telco Customer Churn dataset.

After preprocessing:

| Property | Value |
|---|---:|
| Original observations | 7,043 |
| Clean observations | 7,032 |
| Removed observations | 11 |
| Number of features | 45 |
| Non-churn observations | 5,163 |
| Churn observations | 1,869 |

The target variable is binary:

- `0` = No Churn
- `1` = Churn

The dataset contains customer information related to services, contracts, billing, tenure, and other customer characteristics.

---

## Data Preprocessing

Data preprocessing is performed using Apache Spark.

The preprocessing pipeline includes:

1. Loading the raw CSV dataset
2. Cleaning blank and invalid values
3. Converting numerical attributes to appropriate data types
4. Removing incomplete observations
5. Encoding categorical variables using `StringIndexer`
6. Applying one-hot encoding using `OneHotEncoder`
7. Combining predictors using `VectorAssembler`
8. Converting the churn target into a numeric label
9. Saving the processed dataset in Parquet format

The final processed dataset contains a Spark ML feature vector with 45 features.

Using Parquet allows the Spark feature-vector representation to be preserved and loaded directly during model training.

---

## Machine-Learning Models

Four Spark MLlib classification algorithms are evaluated.

### Logistic Regression

Configuration:

```text
maxIter = 50
```

### Decision Tree

Configuration:

```text
maxDepth = 5
seed = 42
```

### Random Forest

Configuration:

```text
numTrees = 50
maxDepth = 8
seed = 42
```

### Gradient Boosted Tree

Configuration:

```text
maxIter = 30
seed = 42
```

---

## Train/Test Design

The cleaned dataset was divided into training and testing partitions using an approximately 80/20 split with seed 42.

| Split | Rows | Percentage |
|---|---:|---:|
| Training | 5,690 | 80.92% |
| Testing | 1,342 | 19.08% |
| Total | 7,032 | 100% |

Predictive conclusions are based on the held-out test set.

The training and testing partitions were kept separate during the scalability experiment.

---

## Evaluation Metrics

### Predictive Performance

Predictive performance is evaluated using:

- Accuracy
- Churn-class Precision
- Churn-class Recall
- Churn-class F1 Score
- Weighted F1 Score
- ROC-AUC
- Confusion Matrix

### Computational Performance

Computational performance is evaluated using:

- Training time
- Prediction time
- Training throughput
- Prediction throughput
- Timing standard deviation

---

## Baseline Predictive Results

The four models were evaluated on the same held-out test set.

| Model | Accuracy | Churn Precision | Churn Recall | Churn F1 | Weighted F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8219 | 0.7016 | 0.5912 | 0.6417 | 0.8168 | 0.8549 |
| Decision Tree | 0.7951 | 0.6883 | 0.4392 | 0.5363 | 0.7789 | 0.7473 |
| Random Forest | 0.8115 | 0.7243 | 0.4862 | 0.5818 | 0.7983 | 0.8532 |
| Gradient Boosted Tree | 0.8025 | 0.6633 | 0.5442 | 0.5979 | 0.7960 | 0.8434 |

Logistic Regression achieved the strongest overall baseline predictive performance.

Its main results were:

- Accuracy: **82.19%**
- ROC-AUC: **85.49%**
- Churn-class F1: **64.17%**
- Churn recall: **59.12%**
- Churn precision: **70.16%**

Random Forest achieved the highest churn precision at **72.43%**.

---

## Baseline Confusion Matrices

### Logistic Regression

| | Predicted No Churn | Predicted Churn |
|---|---:|---:|
| Actual No Churn | 889 | 91 |
| Actual Churn | 148 | 214 |

### Decision Tree

| | Predicted No Churn | Predicted Churn |
|---|---:|---:|
| Actual No Churn | 908 | 72 |
| Actual Churn | 203 | 159 |

### Random Forest

| | Predicted No Churn | Predicted Churn |
|---|---:|---:|
| Actual No Churn | 913 | 67 |
| Actual Churn | 186 | 176 |

### Gradient Boosted Tree

| | Predicted No Churn | Predicted Churn |
|---|---:|---:|
| Actual No Churn | 880 | 100 |
| Actual Churn | 165 | 197 |

---

## Baseline Computational Results

The baseline training run also recorded execution times for the four models.

These values represent a single execution and can vary between runs depending on JVM state, caching, operating-system activity, and other runtime conditions.

The repeated scalability experiment below is therefore used as the stronger computational comparison.

---

## Scalability Experiment

Computational scalability was evaluated using six workload sizes.

| Scale | Total Rows | Training Rows | Testing Rows |
|---|---:|---:|---:|
| 1x | 7,032 | 5,690 | 1,342 |
| 5x | 35,160 | 28,450 | 6,710 |
| 10x | 70,320 | 56,900 | 13,420 |
| 20x | 140,640 | 113,800 | 26,840 |
| 50x | 351,600 | 284,500 | 67,100 |
| 100x | 703,200 | 569,000 | 134,200 |

The training and testing partitions were replicated separately so that the original train/test separation was preserved.

The experiment included Spark/JVM warm-up before the measured trials.

Each model and workload-size combination was measured three times.

The complete experiment therefore contained:

```text
6 workload sizes × 4 models × 3 trials = 72 timing trials
```

This produced 24 model/workload summaries.

Mean execution time and sample standard deviation were calculated from the repeated measurements.

The scalability experiment also supports checkpointing and resuming. Successful trials are temporarily saved during execution so that completed work is not lost if a long-running Spark/JVM process terminates unexpectedly.

---

## Largest Workload Comparison

At the 100x workload, the experiment processed:

- **569,000 training rows**
- **134,200 testing rows**
- **703,200 total workload rows**

The final repeated results at this workload were:

| Model | Accuracy | ROC-AUC | Mean Training Time (s) | Training SD (s) | Mean Prediction Time (s) |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8219 | 0.8549 | 16.5364 | 1.8987 | 0.5039 |
| Decision Tree | 0.7936 | 0.7455 | 16.1523 | 1.9169 | 0.6754 |
| Random Forest | 0.8145 | 0.8529 | 76.2010 | 13.5178 | 2.2469 |
| Gradient Boosted Tree | 0.8137 | 0.8450 | 92.9097 | 11.4308 | 1.4822 |

At the largest workload, Logistic Regression maintained strong predictive performance while requiring substantially less training time than Random Forest and Gradient Boosted Tree.

Decision Tree had a similar mean training time to Logistic Regression at 100x, but its predictive performance was weaker.

The predictive metrics reported for replicated workloads are treated only as consistency checks. They are not interpreted as evidence of improved predictive generalization because the additional rows are replicated observations.

---

## Main Finding

The experiments indicate that **Logistic Regression provides the best overall balance between predictive performance and computational scalability** among the four models tested in this project.

Logistic Regression achieved the highest baseline:

- Accuracy
- Churn-class F1
- Weighted F1
- ROC-AUC

Random Forest achieved the highest churn-class precision and a similar ROC-AUC, but required considerably more training time as workload size increased.

At the 100x workload, Logistic Regression required approximately **16.54 seconds** of mean training time compared with approximately **76.20 seconds** for Random Forest and **92.91 seconds** for Gradient Boosted Tree.

These findings apply to the experimental environment used in this project. They should not be interpreted as evidence that Logistic Regression will always outperform the other models on different datasets, hardware configurations, or distributed environments.

---

## Generated Visualizations

The visualization pipeline generates 14 research figures:

1. Baseline predictive metrics comparison
2. Model accuracy comparison
3. Model ROC-AUC comparison
4. Churn-class F1 comparison
5. Baseline training-time comparison
6. Baseline prediction-time comparison
7. Training-time scalability
8. Training-time scalability with standard-deviation error bars
9. Prediction-time scalability
10. Prediction-time scalability with standard-deviation error bars
11. Training-throughput scalability
12. Prediction-throughput scalability
13. 100x training-time comparison
14. 100x prediction-time comparison

The generated figures are stored in:

```text
results/figures/
```

---

## Project Structure

```text
Scalable_ML_Big_Data_Project/
│
├── data/
│   ├── Telco-Customer-Churn.csv
│   └── processed/
│       └── churn_data.parquet/
│
├── models/
│   ├── Logistic_Regression/
│   ├── Decision_Tree/
│   ├── Random_Forest/
│   └── Gradient_Boosted_Tree/
│
├── results/
│   ├── model_comparison.csv
│   ├── scalability_trials.csv
│   ├── scalability_repeated_results.csv
│   ├── churn_predictions/
│   └── figures/
│
├── src/
│   ├── load_data.py
│   ├── data_preprocessing.py
│   ├── train_model.py
│   ├── predict.py
│   ├── scalability_experiment.py
│   └── visualize_results.py
│
├── README.md
├── requirements.txt
└── .gitignore
```

Generated Python `__pycache__/` directories are excluded from version control through `.gitignore`.

---

## Installation

Python 3 and a Java runtime compatible with the installed PySpark version are required.

Install the Python dependencies from the project root:

```bash
pip install -r requirements.txt
```

The current `requirements.txt` contains:

```text
pyspark
matplotlib
numpy
```

---

## How to Run the Project

Run the following commands from the project root.

### 1. Inspect the Raw Dataset

Place the Telco Customer Churn CSV file at:

```text
data/Telco-Customer-Churn.csv
```

Then optionally inspect the dataset using:

```bash
python src/load_data.py
```

### 2. Run Data Preprocessing

```bash
python src/data_preprocessing.py
```

This cleans and transforms the raw data and creates:

```text
data/processed/churn_data.parquet/
```

### 3. Train and Evaluate the Models

```bash
python src/train_model.py
```

This trains all four classifiers, evaluates them using the held-out test set, saves the trained models, and creates:

```text
results/model_comparison.csv
```

### 4. Generate Churn Predictions

```bash
python src/predict.py
```

The prediction script loads the selected trained model and applies it to the processed dataset.

The resulting predictions are written to:

```text
results/churn_predictions/
```

The prediction output is intended as an inference demonstration. Predictive performance conclusions are based on the held-out test evaluation performed by `train_model.py`.

### 5. Run the Scalability Experiment

```bash
python src/scalability_experiment.py
```

This performs the repeated benchmark across all six workload sizes and four algorithms.

A complete run contains 72 measured trials.

During a run, temporary checkpoint CSV files may be created so that an interrupted experiment can be resumed.

After successful completion, the final results are written to:

```text
results/scalability_trials.csv
results/scalability_repeated_results.csv
```

### 6. Generate Visualizations

```bash
python src/visualize_results.py
```

This reads the baseline and repeated scalability results and generates all 14 figures in:

```text
results/figures/
```

---

## Generated Outputs

Important experiment outputs include:

- `results/model_comparison.csv` — held-out baseline model comparison
- `results/scalability_trials.csv` — raw repeated timing trials
- `results/scalability_repeated_results.csv` — aggregated scalability results
- `results/figures/` — generated research visualizations
- `results/churn_predictions/` — generated churn predictions
- `models/` — saved Spark MLlib models
- `data/processed/churn_data.parquet/` — processed Spark dataset

Some generated artifacts may be excluded from Git version control because they can be reproduced by running the source code.

---

## Reproducibility Notes

The project uses:

- A fixed train/test split seed of `42`
- Fixed model configurations
- The same base training and testing partitions across scalability workloads
- Three measured trials for each model/workload combination
- Spark/JVM warm-up before measured scalability trials
- Mean and sample standard deviation for repeated timing measurements

Computational timing results are environment-dependent. Exact execution times may differ across machines or across repeated executions on the same machine.

---

## Limitations

This project has several limitations:

- The cleaned dataset contains only 7,032 independent customer observations.
- Larger workloads were generated through replication and do not represent additional independent customers.
- The replicated workloads were created only for computational scalability testing.
- Spark was executed using `local[2]` on a single machine.
- The experiment therefore does not demonstrate multi-node cluster scalability.
- Only four machine-learning algorithms were evaluated.
- Fixed hyperparameter configurations were used.
- No dedicated class-balancing technique was evaluated.
- Computational timings can be affected by hardware, JVM state, caching, operating-system activity, and Spark scheduling.

---

## Future Work

Possible future improvements include:

- Testing on genuinely large datasets containing independent observations
- Running the experiments on a multi-node Spark cluster
- Performing systematic hyperparameter optimization
- Applying cross-validation
- Investigating class weighting and resampling
- Performing classification-threshold optimization to improve churn recall
- Adding feature-importance and model-interpretability analysis
- Measuring memory consumption and end-to-end pipeline latency
- Comparing Spark MLlib with additional scalable machine-learning frameworks

---

## Author

**Irshad Ahmed**

Research Project:

**Scalable Customer Churn Prediction Using Big Data and Machine Learning**

Technologies: Python, PySpark, Apache Spark MLlib, NumPy, Matplotlib