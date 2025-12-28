import os
import zipfile
import glob
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes

# 1. Connect to Azure
credential = DefaultAzureCredential()
ml_client = MLClient.from_config(credential=credential)

# 2. Setup Paths
data_path = "./data/raw"
os.makedirs(data_path, exist_ok=True)

# 3. Download Data (Without --unzip to avoid CLI errors)
print(f"⬇️ Downloading Home Credit data to {data_path}...")
exit_code = os.system(f"kaggle competitions download -c home-credit-default-risk -p {data_path}")

if exit_code != 0:
    print("❌ Kaggle download failed. Check your API key and internet connection.")
    exit(1)

# 4. Manually Unzip (The Safe Way)
zip_files = glob.glob(os.path.join(data_path, "*.zip"))
if zip_files:
    print(f"📦 Found {len(zip_files)} zip file(s). Unzipping...")
    for zip_file in zip_files:
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(data_path)
        print(f"   Extracted: {zip_file}")
        os.remove(zip_file) # Clean up to save space
else:
    print("⚠️ No zip files found. Checking if files are already there...")

# Verify files exist before uploading
if not os.listdir(data_path):
    print("❌ Error: Directory is still empty. Download failed completely.")
    exit(1)

# 5. Register Data to Azure
print("☁️ Registering data asset in Azure...")
my_data = Data(
    path=data_path,
    type=AssetTypes.URI_FOLDER,
    description="Raw Home Credit Default Risk Data (7 Tables)",
    name="home-credit-raw",
    version="1.0"
)

ml_client.data.create_or_update(my_data)
print("✅ Success! Data is registered. Check the 'Data' tab in AML Studio.")