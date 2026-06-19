# ---LINK to the video: https://youtu.be/4Omz3CK24bo

import os
import json
from datetime import date

import requests
from dotenv import load_dotenv
from openai import OpenAI
from prefect import flow, task
from azure.identity import DefaultAzureCredential
from azure.storage.blob import ContainerClient

# ---ETL Pipeline---
# load environment variables from .env file
load_dotenv()

ACCOUNT_URL = "https://merimctd2026sa.blob.core.windows.net"
CONTAINER = "pipeline-data"
MAX_RECORDS = 24
LATITUDE = 34.1659
LONGITUDE = -118.6103


SYSTEM_PROMPT = (
    "You are classifying hourly weather conditions for outdoor running. "
    "Given a temperature in Celsius and a precipitation amount in mm, "
    "classify the conditions as exactly one of: good, marginal, or bad. "
    "Reply with that one word only -- no punctuation, no explanation."
)

VALID_LABELS = {"good", "marginal", "bad"}

# Extract task

# The extract task retrieves weather forecast data from the Open-Meteo API for a given latitude and longitude. It makes an HTTP GET request, checks for errors, and returns the JSON response as a dictionary.
@task(retries=2, retry_delay_seconds=10)
def extract(latitude: float, longitude: float) -> dict:
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={latitude}&longitude={longitude}"
        "&hourly=temperature_2m,precipitation"
        "&forecast_days=7"
    )

    # Make the API request and check for errors. If the request fails, Prefect will automatically retry it up to 2 times with a 10-second delay between attempts.
    response = requests.get(url)
    response.raise_for_status()

    print(f"Extracted forecast data for ({latitude}, {longitude})")

    # The response is returned as a dictionary parsed from the JSON content of the API response.
    return response.json()


def make_user_message(record: dict) -> str:
    return (
        f"Temperature: {record['temperature_2m']}C, "
        f"Precipitation: {record['precipitation']}mm"
    )

# Transform task

# The transform task enriches the raw weather data by classifying the conditions for outdoor running using an OpenAI model. It processes each hourly record, sends it to the model, and appends the classification label to create enriched records.
@task
def transform(data: dict, max_records: int) -> list:

    # The OpenAI client is initialized using the API key from the environment variable. This client is used to send requests to the OpenAI API for classifying the weather conditions.
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    # The task iterates through the hourly weather data, constructs a user message for each record, and sends it to the OpenAI model. The model's response is parsed to extract the classification label, which is added to the enriched records.
    hourly = data["hourly"]

    records = []
    for i in range(min(max_records, len(hourly["time"]))):
        records.append({
            "time": hourly["time"][i],
            "temperature_2m": hourly["temperature_2m"][i],
            "precipitation": hourly["precipitation"][i],
        })

    enriched = []

    # For each record, the task constructs a user message with the temperature and precipitation, sends it to the OpenAI model, and processes the response to classify the conditions. If the classification fails for any reason (e.g., API error), it logs the error and assigns an "unknown" label.
    
    for i, record in enumerate(records):
        user_msg = make_user_message(record)

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
            )

            raw_label = response.choices[0].message.content.strip().lower()
            label = raw_label if raw_label in VALID_LABELS else "unknown"

        except Exception as e:
            print(f"  Classification failed for record {i + 1}: {e}")
            label = "unknown"


        enriched.append({**record, "conditions": label})

        if (i + 1) % 6 == 0:
            print(f"  Classified {i + 1}/{len(records)} records")

    print(f"Transform complete: {len(enriched)} records enriched")
    return enriched


# Load task

# The load task uploads the enriched records to Azure Blob Storage. It converts the records to JSON, encodes them as bytes, and uses the Azure SDK to upload the data to a specified blob path. The overwrite=True parameter allows it to replace existing files at that path if necessary.
@task
def load(records: list, blob_path: str) -> None:
    credential = DefaultAzureCredential()

    container = ContainerClient(
        ACCOUNT_URL,
        CONTAINER,
        credential=credential
    )
    # Convert the records list into JSON, then encode it as UTF-8 bytes so it can be uploaded to Blob Storage.
    payload = json.dumps(records).encode("utf-8")

    container.upload_blob(
        blob_path,
        payload,
        overwrite=True
    )

    print(f"Loaded {len(payload)} bytes to {blob_path}")


# Main flow

# The etl_pipeline flow orchestrates the execution of the extract, transform, and load tasks. It defines the order of operations and handles the data flow between tasks. 
@flow(log_prints=True)
def etl_pipeline(
    latitude: float = LATITUDE,
    longitude: float = LONGITUDE
):
    today = date.today().isoformat()
    blob_path = f"final/{today}/weather_etl.json"

    data = extract(latitude, longitude)
    enriched = transform(data, max_records=MAX_RECORDS)
    load(enriched, blob_path)

    print(f"Pipeline complete. Results at {blob_path}")

if __name__ == "__main__":
    etl_pipeline()
