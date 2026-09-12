from pathlib import Path

from pyspark.sql import SparkSession


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_PATH = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_PATH
    / "data"
    / "Telco-Customer-Churn.csv"
)


# ============================================================
# CREATE SPARK SESSION
# ============================================================

spark = (
    SparkSession.builder
    .appName("TelcoCustomerChurn")
    .master("local[2]")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")


# ============================================================
# LOAD DATASET
# ============================================================

print("=" * 60)
print("TELCO CUSTOMER CHURN - DATA LOADING")
print("=" * 60)

print("\nDataset path:")
print(DATA_PATH)


df = spark.read.csv(
    str(DATA_PATH),
    header=True,
    inferSchema=True
)


# ============================================================
# DATASET INFORMATION
# ============================================================

print("\nTotal Rows:")
print(df.count())

print("\nColumns:")
print(df.columns)

print("\nData Structure:")
df.printSchema()

print("\nFirst 5 Records:")
df.show(
    5,
    truncate=False
)


# ============================================================
# STOP SPARK
# ============================================================

spark.stop()

print("\nData loading completed successfully.")