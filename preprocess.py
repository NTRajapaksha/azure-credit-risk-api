import pandas as pd
import numpy as np
import gc
import os
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

# 1. Setup - Optimize for Small VMs (The "Senior" Trick)
# We force data types to float32 to save 50% RAM.
def optimize_floats(df):
    for col in df.columns:
        if df[col].dtype == 'float64':
            df[col] = df[col].astype('float32')
    return df

def main():
    print("🚀 Starting Feature Engineering Job...")
    
    # 2. Define Paths (Azure Standard Paths)
    # We assume data is downloaded in ./data/raw from the previous step
    base_path = "./data/raw"
    output_path = "./data/processed"
    os.makedirs(output_path, exist_ok=True)

    # 3. Process Bureau Balance (The Deepest Table)
    print("   Processing Bureau Balance...")
    bureau_balance = pd.read_csv(f"{base_path}/bureau_balance.csv")
    bureau_balance = optimize_floats(bureau_balance)
    
    # One-Hot Encoding for 'STATUS' (Paid, Late, Written Off)
    bb_cat = pd.get_dummies(bureau_balance[['STATUS', 'SK_ID_BUREAU']], columns=['STATUS'])
    
    # Aggregate by Loan ID (SK_ID_BUREAU)
    bb_agg = bb_cat.groupby('SK_ID_BUREAU').mean()
    
    # Clean up RAM immediately
    del bureau_balance, bb_cat
    gc.collect()

    # 4. Process Bureau Data (The User's History)
    print("   Processing Bureau Data...")
    bureau = pd.read_csv(f"{base_path}/bureau.csv")
    bureau = optimize_floats(bureau)
    
    # Join with the Balance aggregates we just created
    bureau = bureau.merge(bb_agg, on='SK_ID_BUREAU', how='left')
    del bb_agg
    gc.collect()
    
    # 5. Create "Senior" Aggregates (The Domain Knowledge)
    # We want to know: "What is the MAX days this person has been overdue across ALL loans?"
    # We group by the CURRENT Applicant ID (SK_ID_CURR)
    
    aggregations = {
        'DAYS_CREDIT': ['min', 'max', 'mean'],
        'DAYS_CREDIT_ENDDATE': ['min', 'max'],
        'AMT_CREDIT_SUM': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_DEBT': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_OVERDUE': ['mean', 'max'],
        # Add the status columns we created earlier
        'STATUS_0': ['mean'], # Avg percent of time they are on time
        'STATUS_5': ['mean']  # Avg percent of time they are hopelessly late
    }
    
    print("   Aggregating Bureau History per Applicant...")
    bureau_agg = bureau.groupby('SK_ID_CURR').agg(aggregations)
    
    # Flatten the Multi-Index columns (e.g. DAYS_CREDIT_min, DAYS_CREDIT_max)
    bureau_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in bureau_agg.columns.tolist()])
    
    # 6. Save to Parquet (Fast & Small)
    print(f"💾 Saving {bureau_agg.shape[0]} feature rows to {output_path}...")
    bureau_agg.reset_index().to_parquet(f"{output_path}/bureau_features.parquet")
    
    print("✅ Feature Engineering Complete. File saved: bureau_features.parquet")

if __name__ == "__main__":
    main()