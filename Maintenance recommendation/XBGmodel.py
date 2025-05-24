from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.sql.functions import col
from pyspark.sql.types import DoubleType, IntegerType, StructField, StructType, StringType, TimestampType
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix
import time
import os
import sys

# Set environment variables for Windows
os.environ['HADOOP_HOME'] = "C:\\hadoop\\hadoop-3.3.6"
os.environ['PATH'] = f"{os.environ['HADOOP_HOME']}\\bin;{os.environ.get('PATH', '')}"
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

# Create temp directory if it doesn't exist
temp_dir = os.path.join(os.path.expanduser("~"), "spark_temp")
os.makedirs(temp_dir, exist_ok=True)

# Initialize Spark session with Windows configurations
spark = SparkSession.builder \
    .appName("XGBoostAlertPrediction") \
    .config("spark.driver.memory", "8g") \
    .config("spark.executor.memory", "8g") \
    .config("spark.driver.maxResultSize", "4g") \
    .config("spark.local.dir", temp_dir) \
    .config("spark.sql.warehouse.dir", os.path.join(temp_dir, "warehouse")) \
    .config("spark.io.compression.codec", "snappy") \
    .config("spark.rdd.compress", "false") \
    .config("spark.sql.broadcastTimeout", "600") \
    .master("local[*]") \
    .getOrCreate()

# Your schema definition remains the same
def define_schema():
    return StructType([
        # ... (keep your existing schema definition)
    ])

try:
    # Load the data with alerts using your specific path
    schema = define_schema()
    data = spark.read.csv(
        r"C:/Users/Son Phan/Scalable/Predictive-Maintenance-System-using-Apache-Spark/Maintenance recommendation/dataset_maintenance_recommendation.csv",
        header=True,
        schema=schema
    )

    # Print available columns
    print("Available columns in the dataset:")
    data.printSchema()
    print("\nFirst few rows of the data:")
    data.show(5)

    # Define feature columns based on the actual columns in your dataset
    feature_cols = [
        "temperature", "vibration", "pressure", "rotational_speed", "power_output",
        "noise_level", "voltage", "current", "oil_viscosity", "production_rate",
        "operating_hours", "downtime_hours", "ambient_temperature", "ambient_humidity",
        "days_since_maintenance", "equipment_age_days", "expected_lifetime_years",
        "warranty_period_years", "cumulative_maintenance_cost"
    ]

    # Verify that all feature columns exist in the dataset
    available_columns = data.columns
    missing_columns = [col for col in feature_cols if col not in available_columns]
    
    if missing_columns:
        print("\nWarning: The following columns are missing from the dataset:")
        print(missing_columns)
        print("\nAvailable columns are:")
        print(available_columns)
        raise ValueError("Some feature columns are missing from the dataset")

    # Create feature vector
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    data = assembler.transform(data)

    # Create label index
    label_indexer = StringIndexer(inputCol="maintenance_needed", outputCol="label")
    data = label_indexer.fit(data).transform(data)

    # Convert Spark DataFrame to pandas DataFrame
    pandas_df = data.select("features", "label").toPandas()

    # Extract features and labels
    X = np.array(pandas_df['features'].tolist())
    y = pandas_df['label'].values

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create DMatrix for XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    # Set XGBoost parameters
    params = {
        'max_depth': 6,
        'eta': 0.3,
        'objective': 'multi:softprob',
        'num_class': 2  # Changed to 2 since maintenance_needed has 2 classes (Maintenance required/No maintenance required)
    }

    # Train the model
    num_round = 100
    model = xgb.train(params, dtrain, num_round)

    # Make predictions
    preds = model.predict(dtest)
    best_preds = np.argmax(preds, axis=1)

    # Print classification report and confusion matrix
    print("Classification Report:")
    print(classification_report(y_test, best_preds))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, best_preds))

    # Feature importance
    importance = model.get_score(importance_type='weight')
    feature_importance = [(feature_cols[int(feature.replace('f', ''))], score) for feature, score in importance.items()]
    sorted_importance = sorted(feature_importance, key=lambda x: x[1], reverse=True)
    print("\nTop 5 Important Features:")
    for feature, score in sorted_importance[:5]:
        print(f"{feature}: {score}")

    # Example prediction
    print("\nExample Prediction:")
    random_index = np.random.randint(0, len(X_test))
    sample = X_test[random_index]
    true_label = y_test[random_index]

    sample_dmatrix = xgb.DMatrix([sample])
    prediction = model.predict(sample_dmatrix)[0]
    predicted_class = np.argmax(prediction)

    print(f"True label: {true_label}")
    print(f"Predicted probabilities: {prediction}")
    print(f"Predicted class: {predicted_class}")

    print("\nFeature values for this sample:")
    for feature, value in zip(feature_cols, sample):
        print(f"{feature}: {value:.2f}")

except Exception as e:
    print("An error occurred:", str(e))
finally:
    # Stop Spark session
    spark.stop()
    print("Process completed!")
