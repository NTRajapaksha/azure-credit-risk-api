import pandas as pd
import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import shap
import joblib
import os
import gc
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# 1. Settings
RAW_PATH = "./data/raw"
PROCESSED_PATH = "./data/processed"
MODEL_DIR = "./model_output"
os.makedirs(MODEL_DIR, exist_ok=True)

def main():
    # Start an MLflow experiment (Tracks your work in Azure UI)
    mlflow.set_experiment("home-credit-risk-v1")
    
    with mlflow.start_run():
        print("🚀 Loading Data...")
        
        # Load Main Application Data
        df = pd.read_csv(f"{RAW_PATH}/application_train.csv")
        
        # Load the Bureau Features we created in preprocess.py
        print("   Merging Bureau Features...")
        bureau_feats = pd.read_parquet(f"{PROCESSED_PATH}/bureau_features.parquet")
        
        # Left Join: Attach history to the applicant
        df = df.merge(bureau_feats, on='SK_ID_CURR', how='left')
        
        # Clean up RAM
        del bureau_feats
        gc.collect()
        
        # 2. Preprocessing for Training
        print("   Preparing Training Data...")
        
        # Drop ID and Target from features
        X = df.drop(columns=['SK_ID_CURR', 'TARGET'])
        y = df['TARGET']
        
        # Categorical Handling (LightGBM handles this, but we need to convert to 'category' dtype)
        for col in X.columns:
            if X[col].dtype == 'object':
                X[col] = X[col].astype('category')
        
        # Split
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1, random_state=42)
        
        # 3. Train Model
        print("🧠 Training LightGBM Model...")
        
        # Hyperparameters (Optimized for speed on your small VM)
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'n_estimators': 100
        }
        
        # Train
        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            eval_metric='auc',
            callbacks=[lgb.early_stopping(10)]
        )
        
        # 4. Evaluation & Logging
        print("   Logging Metrics to Azure ML...")
        y_pred = model.predict_proba(X_valid)[:, 1]
        auc = roc_auc_score(y_valid, y_pred)
        print(f"   ✅ Validation AUC: {auc:.4f}")
        
        # Log metric to Azure Cloud
        mlflow.log_metric("AUC", auc)
        
        # 5. Save Artifacts (The "Nuclear" Fix)
        print("💾 Saving Model in Native Format...")
        
        # SAVE AS TEXT (Universal format, no pickle/joblib needed)
        model.booster_.save_model(f"{MODEL_DIR}/model.txt")
        
        # We still need to pickle the explainer, but that's usually safer
        print("💾 Saving Explainer...")
        explainer = shap.TreeExplainer(model)
        joblib.dump(explainer, f"{MODEL_DIR}/explainer.pkl")
        
        # Log to MLflow
        mlflow.sklearn.log_model(model, "model")
        
        print("🎉 Training Complete! Saved: model.txt and explainer.pkl")
        

if __name__ == "__main__":
    main()