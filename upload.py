import os
from azure.storage.blob import BlobServiceClient

# --- PASTE YOUR CONNECTION STRING BELOW ---
CONN_STR = "PASTE_YOUR_CONNECTION_STRING_HERE" 

CONTAINER_NAME = "models"
LOCAL_MODEL_DIR = "./model_output"

print("🚀 Connecting using Connection String...")

# 1. Connect using the Key (Bypasses Permission Errors)
try:
    blob_service_client = BlobServiceClient.from_connection_string(CONN_STR)
except Exception as e:
    print(f"❌ Connection failed. Check your string. Error: {e}")
    exit()

# 2. Create/Get Container
try:
    container_client = blob_service_client.create_container(CONTAINER_NAME)
    print(f"   Created container '{CONTAINER_NAME}'")
except Exception:
    container_client = blob_service_client.get_container_client(CONTAINER_NAME)
    print(f"   Found container '{CONTAINER_NAME}'")

# 3. Upload Files
# Changed model.pkl to model.txt
files_to_upload = ["model.txt", "explainer.pkl"]

for filename in files_to_upload:
    file_path = os.path.join(LOCAL_MODEL_DIR, filename)
    
    if not os.path.exists(file_path):
        print(f"❌ Error: {filename} not found in {LOCAL_MODEL_DIR}")
        continue

    print(f"⬆️  Uploading {filename}...")
    blob_client = container_client.get_blob_client(blob=filename)
    
    with open(file_path, "rb") as data:
        blob_client.upload_blob(data, overwrite=True)
    print(f"   ✅ Uploaded: {filename}")

print("🎉 All files uploaded successfully!")