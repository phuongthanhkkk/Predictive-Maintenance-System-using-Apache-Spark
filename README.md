# Predictive Maintenance System using Apache Spark

This repository hosts a **mini-project** focused on building a **Predictive Maintenance System**, leveraging **Apache Spark** for scalable data processing and analysis. The system aims to **predict potential equipment failures** and recommend **maintenance actions**, thereby reducing downtime and optimizing operational efficiency.

---

## Overview

The project is developed as part of a **Scalable-Distributed-Computing** course at **VNUHCM - International University**. It demonstrates an **end-to-end solution** for predictive maintenance, from **data processing** and **model training** to **alert generation** and **user interface presentation**.

---

## Features

- **Data Processing & Analysis**  
  Utilizes Apache Spark for efficient handling and analysis of large datasets related to equipment performance and historical failures.

- **Predictive Modeling**  
  Implements machine learning models (e.g., **XGBoost**) to predict equipment failures.

- **Alert System**  
  Generates alerts based on predicted failures, enabling proactive intervention.

- **Maintenance Recommendation**  
  Provides recommendations for maintenance actions to prevent impending failures.

- **User Interface (UI)**  
  A front-end interface for visualizing data, model predictions, and managing alerts/recommendations.

---

## Technologies Used

- **Apache Spark**: For distributed data processing and analytics  
- **Python**: Primary programming language for development  
- **Jupyter Notebook**: For data exploration, model prototyping, and analysis  
- **XGBoost**: A gradient boosting framework, likely used for the predictive model (`xgb_model.joblib`)

---

## Project Structure

```text
.
├── Alert System/
├── Data Processing & Analysis/
├── Maintenance recommendation/
├── Predictive-Maintenance-System-using-Apache-Spark/
├── UI/
└── xgb_model.joblib
```

- `Predictive-Maintenance-System-using-Apache-Spark/`: Likely contains the core Spark applications and scripts.
- `UI/`: Front-end code for the user interface.
- `xgb_model.joblib`: A pre-trained XGBoost model.

---

## Installation and Setup (High-Level)

To set up and run this project, you will generally need:

- **Apache Spark**: A running Spark cluster (or local setup).
- **Python Environment**: Python 3.x with necessary libraries such as:
  - `pyspark`
  - `pandas`
  - `scikit-learn`
  - `xgboost`
  - `streamlit` (or other UI frameworks if used)

  Detailed installation instructions for specific modules might be found within their respective directories.

---

## Usage

1. **Prepare Data**  
   Ensure your equipment sensor data and historical maintenance records are accessible.

2. **Run Data Processing**  
   Execute Spark jobs within the `Data Processing & Analysis/` directory to prepare data for modeling.

3. **Train/Load Model**  
   Utilize or retrain the predictive model, using:
   - `xgb_model.joblib`, or  
   - Scripts in the `Predictive-Maintenance-System-using-Apache-Spark/` directory.

4. **Run Prediction & Alerting**  
   Execute the prediction and alerting logic to identify potential failures.

5. **Access UI**  
   Launch the user interface to visualize results and interact with the system.

Specific commands and detailed steps would typically be provided within individual module `README.md` or documentation files.

---

## Contributors

- **Nguyen Quang Sang**
- **Phan Manh Son**
- **Bui Phuong Thanh**

