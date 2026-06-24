
### LINK to the video presentation: https://youtu.be/gMs2Cu7qTWY ###

import os
import requests
import json
from datetime import date
import pandas as pd
from azure.storage.blob import ContainerClient
from azure.identity import DefaultAzureCredential

ACCOUNT_URL = "https://merimctd2026sa.blob.core.windows.net"
CONTAINER = "pipeline-data"
LATITUDE = 34.1659
LONGITUDE = -118.6103


# Extract
url = (
    f"https://api.open-meteo.com/v1/forecast"
    f"?latitude={LATITUDE}&longitude={LONGITUDE}"
    f"&hourly=temperature_2m,precipitation"
    f"&forecast_days=7"
)
response = requests.get(url)

response.raise_for_status()

data = response.json()

# Serialize
payload = json.dumps(data).encode("utf-8")

# Load
today = date.today().isoformat()
blob_path = f"raw/{today}/weather.json"

credential = DefaultAzureCredential()
container = ContainerClient(ACCOUNT_URL, CONTAINER, credential=credential)
container.upload_blob(blob_path, payload, overwrite=True)
print(f"Uploaded {len(payload)} bytes to {blob_path}")

# Verify: list blobs in the container
print("\nBlobs in container:")
for blob in container.list_blobs():
    print(f"  {blob.name}  ({blob.size} bytes)")

# Read back and confirm
raw = container.download_blob(blob_path).readall()
df = pd.DataFrame(json.loads(raw.decode("utf-8"))["hourly"])
print(f"\nFirst 5 rows:")
print(df.head())

# Save downloaded JSON file in outputs
os.makedirs("outputs", exist_ok=True)

with open("./outputs/weather_raw.json", "wb") as f:
    f.write(raw)
