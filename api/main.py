from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import shap

app = FastAPI(title="Predictive Maintenance API")

# Load models once at startup
xgb_model = joblib.load("../models/xgb_model.pkl")
iso_forest = joblib.load("../models/iso_forest.pkl")
explainer = shap.TreeExplainer(xgb_model)


class SensorReading(BaseModel):
    air_temperature_k: float
    process_temperature_k: float
    rotational_speed_rpm: float
    torque_nm: float
    tool_wear_min: float
    type_l: int  # 1 if Type == 'L', else 0
    type_m: int  # 1 if Type == 'M', else 0


@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/predict")
def predict(reading: SensorReading):
    high_torque_flag = int(reading.torque_nm > 46)
    speed_deviation = abs(reading.rotational_speed_rpm - 1503)
    high_torque_and_wear = int(high_torque_flag == 1 and reading.tool_wear_min > 165)

    input_df = pd.DataFrame([{
        "Air temperature K": reading.air_temperature_k,
        "Process temperature K": reading.process_temperature_k,
        "Rotational speed rpm": reading.rotational_speed_rpm,
        "Torque Nm": reading.torque_nm,
        "Tool wear min": reading.tool_wear_min,
        "high_torque_flag": high_torque_flag,
        "speed_deviation": speed_deviation,
        "high_torque_and_wear": high_torque_and_wear,
        "Type_L": reading.type_l,
        "Type_M": reading.type_m,
    }])

    prediction = int(xgb_model.predict(input_df)[0])
    probability = float(xgb_model.predict_proba(input_df)[0][1])

    # SHAP explanation for this specific prediction
    shap_vals = explainer.shap_values(input_df)[0]
    feature_impacts = dict(zip(input_df.columns, shap_vals))
    top_reasons = sorted(feature_impacts.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
    top_reasons_formatted = [
        {"feature": feat, "impact": round(float(val), 4)} for feat, val in top_reasons
    ]

    return {
        "failure_prediction": prediction,
        "failure_probability": round(probability, 4),
        "top_reasons": top_reasons_formatted
    }