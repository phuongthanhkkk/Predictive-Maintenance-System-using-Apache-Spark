import streamlit as st
import numpy as np
import os
import pandas as pd
import plotly.express as px
import xgboost as xgb
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.sql.types import DoubleType, IntegerType, StructField, StructType, StringType, TimestampType

# Initialize Spark session
@st.cache_resource
def init_spark():
    return SparkSession.builder \
        .appName("XGBoostAlertPrediction") \
        .config("spark.driver.memory", "8g") \
        .config("spark.executor.memory", "8g") \
        .config("spark.driver.maxResultSize", "4g") \
        .getOrCreate()

def define_schema():
    return StructType([
        StructField("equipment_id", IntegerType(), nullable=True),
        StructField("timestamp", TimestampType(), nullable=True),
        StructField("temperature", DoubleType(), nullable=True),
        StructField("vibration", DoubleType(), nullable=True),
        StructField("pressure", DoubleType(), nullable=True),
        StructField("rotational_speed", DoubleType(), nullable=True),
        StructField("power_output", DoubleType(), nullable=True),
        StructField("noise_level", DoubleType(), nullable=True),
        StructField("voltage", DoubleType(), nullable=True),
        StructField("current", DoubleType(), nullable=True),
        StructField("oil_viscosity", DoubleType(), nullable=True),
        StructField("model", StringType(), nullable=True),
        StructField("manufacturer", StringType(), nullable=True),
        StructField("installation_date", TimestampType(), nullable=True),
        StructField("max_temperature", DoubleType(), nullable=True),
        StructField("max_pressure", DoubleType(), nullable=True),
        StructField("max_rotational_speed", DoubleType(), nullable=True),
        StructField("expected_lifetime_years", DoubleType(), nullable=True),
        StructField("warranty_period_years", IntegerType(), nullable=True),
        StructField("last_major_overhaul", TimestampType(), nullable=True),
        StructField("location", StringType(), nullable=True),
        StructField("criticality", StringType(), nullable=True),
        StructField("maintenance_type", StringType(), nullable=True),
        StructField("description", StringType(), nullable=True),
        StructField("technician_id", IntegerType(), nullable=True),
        StructField("duration_hours", DoubleType(), nullable=True),
        StructField("cost", DoubleType(), nullable=True),
        StructField("parts_replaced", StringType(), nullable=True),
        StructField("maintenance_result", StringType(), nullable=True),
        StructField("maintenance_date", TimestampType(), nullable=True),
        StructField("production_rate", DoubleType(), nullable=True),
        StructField("operating_hours", DoubleType(), nullable=True),
        StructField("downtime_hours", DoubleType(), nullable=True),
        StructField("operator_id", IntegerType(), nullable=True),
        StructField("product_type", StringType(), nullable=True),
        StructField("raw_material_quality", StringType(), nullable=True),
        StructField("ambient_temperature", DoubleType(), nullable=True),
        StructField("ambient_humidity", DoubleType(), nullable=True),
        StructField("operation_date", TimestampType(), nullable=True),
        StructField("days_since_maintenance", IntegerType(), nullable=True),
        StructField("equipment_age_days", IntegerType(), nullable=True),
        StructField("days_since_overhaul", IntegerType(), nullable=True),
        StructField("temp_pct_of_max", DoubleType(), nullable=True),
        StructField("pressure_pct_of_max", DoubleType(), nullable=True),
        StructField("speed_pct_of_max", DoubleType(), nullable=True),
        StructField("cumulative_maintenance_cost", DoubleType(), nullable=True),
        StructField("cumulative_operating_hours", DoubleType(), nullable=True),
        StructField("estimated_rul", DoubleType(), nullable=True),
        StructField("criticality_encoded", DoubleType(), nullable=True),
        StructField("maintenance_type_encoded", DoubleType(), nullable=True),
        StructField("maintenance_result_encoded", DoubleType(), nullable=True),
        StructField("product_type_encoded", DoubleType(), nullable=True),
        StructField("raw_material_quality_encoded", DoubleType(), nullable=True),
        StructField("parts_replaced_encoded", DoubleType(), nullable=True),
        StructField("temperature_alert", StringType(), nullable=True),
        StructField("pressure_alert", StringType(), nullable=True),
        StructField("rotational_speed_alert", StringType(), nullable=True),
        StructField("power_output_alert", StringType(), nullable=True),
        StructField("noise_level_alert", StringType(), nullable=True),
        StructField("voltage_alert", StringType(), nullable=True),
        StructField("current_alert", StringType(), nullable=True),
        StructField("oil_viscosity_alert", StringType(), nullable=True),
        StructField("alert", StringType(), nullable=True),
        StructField("alert_score", IntegerType(), nullable=True),
        StructField("age_condition_score", IntegerType(), nullable=False),
        StructField("downtime_condition_score", IntegerType(), nullable=False),
        StructField("maintenance_condition_score", IntegerType(), nullable=False),
        StructField("environment_condition_score", IntegerType(), nullable=False),
        StructField("criticality_avg_annual_cost", DoubleType(), nullable=True),
        StructField("threshold", DoubleType(), nullable=True),
        StructField("maintenance_cost_condition_score", IntegerType(), nullable=False),
        StructField("operational_score", IntegerType(), nullable=False),
        StructField("total_score", IntegerType(), nullable=True),
        StructField("maintenance_needed", StringType(), nullable=False),
        StructField("maintenance_item", StringType(), nullable=False)
    ])

# Load and prepare data
@st.cache_data
def load_data():
    spark = init_spark()
    data_path = r"C:\Users\Son Phan\Desktop\maintenance_data.csv"
    
    # Load data with schema
    data = spark.read.csv(data_path, header=True, schema=define_schema())
    data = data.withColumn("maintenance_encoded", 
                          when(col("maintenance_needed") == "Maintenance required", 1).otherwise(0))
    
    # Convert to pandas for easier handling in Streamlit
    return data.toPandas()

# Feature columns (same as in XGB_model.py)
FEATURE_COLS = [
    "temperature", "vibration", "pressure", "rotational_speed", "power_output",
    "noise_level", "voltage", "current", "oil_viscosity", "production_rate",
    "operating_hours", "downtime_hours", "ambient_temperature", "ambient_humidity",
    "days_since_maintenance", "equipment_age_days", "expected_lifetime_years",
    "warranty_period_years", "maintenance_encoded", "maintenance_type_encoded"
]


def show_prediction_page():
    st.header("Predict Maintenance Needs")
    
    # Input form
    with st.form("prediction_form"):
        st.subheader("Equipment Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Sensor Readings")
            temperature = st.number_input("Temperature (°C)", value=25.0)
            vibration = st.number_input("Vibration (mm/s)", value=2.0)
            pressure = st.number_input("Pressure (bar)", value=5.0)
            rotational_speed = st.number_input("Rotational Speed (RPM)", value=1500.0)
            power_output = st.number_input("Power Output (kW)", value=75.0)
            noise_level = st.number_input("Noise Level (dB)", value=65.0)
            voltage = st.number_input("Voltage (V)", value=220.0)
            current = st.number_input("Current (A)", value=10.0)
            oil_viscosity = st.number_input("Oil Viscosity (cSt)", value=32.0)
            
        with col2:
            st.markdown("#### Operational Parameters")
            production_rate = st.number_input("Production Rate", value=0.0)
            operating_hours = st.number_input("Operating Hours", value=1000.0)
            downtime_hours = st.number_input("Downtime Hours", value=0.0)
            ambient_temperature = st.number_input("Ambient Temperature (°C)", value=25.0)
            ambient_humidity = st.number_input("Ambient Humidity (%)", value=50.0)
            days_since_maintenance = st.number_input("Days Since Last Maintenance", value=30)
            equipment_age_days = st.number_input("Equipment Age (Days)", value=365)
            expected_lifetime_years = st.number_input("Expected Lifetime (Years)", value=10.0)
            warranty_period_years = st.number_input("Warranty Period (Years)", value=2)
        
        submitted = st.form_submit_button("Predict")
        
        if submitted:
            try:
                # Create prediction input with all required features
                input_data = np.array([[
                    temperature, vibration, pressure, rotational_speed, power_output,
                    noise_level, voltage, current, oil_viscosity, production_rate,
                    operating_hours, downtime_hours, ambient_temperature, ambient_humidity,
                    days_since_maintenance, equipment_age_days, expected_lifetime_years,
                    warranty_period_years, 
                    0,  # maintenance_encoded (default to 0)
                    0.0  # maintenance_type_encoded (default to 0.0)
                ]])
                
                # Load and use the model
                model_path = r"C:\Users\Son Phan\Scalable\Predictive-Maintenance-System-using-Apache-Spark\xgb_model.joblib"
                if os.path.exists(model_path):
                    print(f"Model file found at: {model_path}")
                    model = xgb.Booster()
                    model.load_model(model_path)
                    print("Model loaded successfully")
                    
                    # Make prediction
                    dmatrix = xgb.DMatrix(input_data)
                    prediction = model.predict(dmatrix)
                    print(f"Raw prediction probabilities: {prediction}")
                    predicted_class = np.argmax(prediction)
                    print(f"Predicted class index: {predicted_class}")
                    
                    # Display results
                    st.subheader("Maintenance Analysis")
                    
                    # Get the actual maintenance items from the data
                    df = load_data()
                    maintenance_items = df['maintenance_item'].unique()
                    maintenance_items = sorted(maintenance_items)  # Sort to ensure consistent order
                    print(f"Available maintenance items: {maintenance_items}")
                    
                    predicted_item = maintenance_items[predicted_class]
                    max_probability = prediction[0][predicted_class]
                    print(f"Selected maintenance item: {predicted_item} with probability: {max_probability}")
                    
                    # Determine if maintenance is needed based on probability threshold
                    maintenance_threshold = 0.3  # You can adjust this threshold
                    needs_maintenance = max_probability > maintenance_threshold
                    
                    # Display maintenance status
                    if needs_maintenance:
                        st.warning("⚠️ Maintenance Required")
                        st.success(f"Recommended Maintenance: {predicted_item}")
                        
                        # Show maintenance details
                        st.subheader("Maintenance Details")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Confidence", f"{max_probability:.1%}")
                            st.metric("Priority", "High" if max_probability > 0.7 else "Medium")
                        with col2:
                            st.metric("Equipment Age", f"{equipment_age_days} days")
                            st.metric("Days Since Last Maintenance", f"{days_since_maintenance} days")
                    else:
                        st.success("✅ No Maintenance Required")
                        st.info("Equipment is operating within normal parameters")
                    
                    # Show probabilities for all maintenance types
                    st.subheader("Maintenance Type Probabilities")
                    prob_df = pd.DataFrame({
                        'Maintenance Type': maintenance_items,
                        'Probability': prediction[0]
                    })
                    # Sort by probability
                    prob_df = prob_df.sort_values('Probability', ascending=False)
                    
                    # Create a color-coded bar chart
                    fig = px.bar(prob_df, 
                                x='Maintenance Type', 
                                y='Probability',
                                color='Probability',
                                color_continuous_scale='RdYlGn',
                                title='Probability of Each Maintenance Type')
                    fig.update_layout(
                        xaxis_title="Maintenance Type",
                        yaxis_title="Probability",
                        yaxis_tickformat='.1%'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show key factors influencing the prediction
                    st.subheader("Key Factors Influencing Prediction")
                    importance = model.get_score(importance_type='weight')
                    feature_importance = [(FEATURE_COLS[int(feature.replace('f', ''))], score) 
                                        for feature, score in importance.items()]
                    sorted_importance = sorted(feature_importance, key=lambda x: x[1], reverse=True)
                    
                    # Create a table of top factors
                    factors_df = pd.DataFrame(sorted_importance[:5], columns=['Factor', 'Importance'])
                    st.table(factors_df)
                    
                else:
                    st.error("Model file not found. Please ensure xgb_model.joblib exists in the current directory.")
                    
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")

def main():
    st.title("🔧 Predictive Maintenance System")
    show_prediction_page()

if __name__ == "__main__":
    main() 