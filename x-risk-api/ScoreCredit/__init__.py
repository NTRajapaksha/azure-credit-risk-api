import logging
import azure.functions as func
import json
import joblib
import numpy as np
import os
import tempfile
import lightgbm as lgb 
from azure.storage.blob import BlobServiceClient

# --- GLOBAL CACHE ---
model = None
explainer = None
# PASTE YOUR CONNECTION STRING HERE
CONN_STR = os.environ["AzureWebJobsStorage"]
CONTAINER_NAME = "models"

def load_artifacts():
    global model, explainer
    logging.info("🚀 Loading artifacts...")
    
    blob_service = BlobServiceClient.from_connection_string(CONN_STR)
    container = blob_service.get_container_client(CONTAINER_NAME)
    
    def download(filename):
        blob = container.get_blob_client(filename)
        tmp = os.path.join(tempfile.gettempdir(), filename)
        with open(tmp, "wb") as f:
            blob.download_blob().readinto(f)
        return tmp

    # 1. Load Model (Native Text - No Joblib Conflicts!)
    model_path = download("model.txt")
    model = lgb.Booster(model_file=model_path)
    
    # 2. Load Explainer (If this fails, we skip it to keep API alive)
    try:
        expl_path = download("explainer.pkl")
        explainer = joblib.load(expl_path)
    except Exception as e:
        logging.warning(f"⚠️ Explainer failed: {e}")
        explainer = None

    logging.info("✅ Artifacts loaded.")

def main(req: func.HttpRequest) -> func.HttpResponse:
    global model, explainer
    
    if model is None:
        try:
            load_artifacts()
        except Exception as e:
            return func.HttpResponse(f"Init Error: {str(e)}", status_code=500)

    try:
        req_body = req.get_json()
        features = np.array(req_body['features']).reshape(1, -1)
        
        # PREDICT (Booster returns raw probability directly)
        prob = model.predict(features)[0]
        decision = "APPROVE" if prob < 0.20 else "REJECT"

        # EXPLAIN
        reasons = []
        if explainer:
            try:
                shap_values = explainer.shap_values(features)
                # Handle SHAP output variations
                sv = shap_values[1][0] if isinstance(shap_values, list) else shap_values[0]
                top_indices = np.argsort(-np.abs(sv))[:3]
                reasons = [f"Feature {i}: {sv[i]:.4f}" for i in top_indices]
            except:
                reasons = ["Explanation unavailable"]

        return func.HttpResponse(json.dumps({
            "decision": decision,
            "risk_score": float(prob),
            "explanation": reasons
        }), mimetype="application/json")

    except Exception as e:
        return func.HttpResponse(f"Inference Error: {str(e)}", status_code=500)
