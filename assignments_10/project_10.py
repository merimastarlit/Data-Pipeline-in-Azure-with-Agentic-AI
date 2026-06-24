
# LINK to the video presentation: https://youtu.be/qk2N5Wh2-qA

# Step 6: Reflection
"""
Reflection:
Classifying weather for outdoor running may not be the best use of an LLM because
the decision can probably be handled with clear rules. A rule-based approach would
be cheaper, faster, and more predictable. The LLM adds flexibility and judgment,
but it may also give inconsistent results. With rules, we gain consistency but lose
some nuance about what “good” running weather means.
"""

import json
import os
from datetime import date
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from azure.storage.blob import ContainerClient
from azure.identity import DefaultAzureCredential


# Constants and Configurations
ACCOUNT_URL = "https://merimctd2026sa.blob.core.windows.net"
CONTAINER = "pipeline-data"

# labels as defined:
VALID_LABELS = {"good", "marginal", "bad"}

SYSTEM_PROMPT = (
    "You are classifying hourly weather conditions for outdoor running. "
    "Given a temperature in Celsius and a precipitation amount in mm, "
    "classify the conditions as exactly one of: good, marginal, or bad. "
    "Reply with that one word only -- no punctuation, no explanation."
)


# Helper Functions
def reshape_weather(data):
    hourly = data["hourly"]
    records = []

    for i in range(len(hourly["time"])):
        records.append({
            "time": hourly["time"][i],
            "temperature_2m": hourly["temperature_2m"][i],
            "precipitation": hourly["precipitation"][i],
        })

    return records


def make_user_message(record):
    return (
        f"Temperature: {record['temperature_2m']}C, "
        f"Precipitation: {record['precipitation']}mm"
    )


# classify_record takes a single weather record and returns the classified conditions
def classify_record(client, record):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": make_user_message(record)},
        ],
    )

    # We expect the response to be a single word: "good", "marginal", or "bad".
    raw_label = response.choices[0].message.content.strip().lower()

    # Validate the label against our expected set. If it's not valid, we can return "unknown" or handle it as needed.
    if raw_label in VALID_LABELS:
        return raw_label

    return "unknown"


# Main Function
def main():
    # Load environment variables (like OPENAI_API_KEY)
    load_dotenv()

    # Step 1: Read the raw weather data from Azure Blob Storage (or fallback to local file)
    today = date.today().isoformat()
    # Define blob paths based on today's date
    raw_blob_path = f"raw/{today}/weather.json"
    processed_blob_path = f"processed/{today}/weather_classified.json"

    # Connect to Azure Blob Storage
    credential = DefaultAzureCredential()
    container = ContainerClient(
        account_url=ACCOUNT_URL,
        container_name=CONTAINER,
        credential=credential,
    )

    # Attempt to load raw data from Blob Storage, with a fallback to a local file if it fails
    try:
        raw = container.download_blob(raw_blob_path).readall()
        data = json.loads(raw.decode("utf-8"))
        print(f"Loaded raw data from Blob Storage: {raw_blob_path}")

    except Exception:
        with open("./assignments_09/outputs/weather_raw.json", "r", encoding="utf-8") as f:
            data = json.load(f)
       
    # Reshape the data into a list of records
    records = reshape_weather(data)
    print(f"Loaded {len(records)} hourly records")

    # Step 2: Transform - classify each record using the LLM
    # Initialize OpenAI client
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    enriched = []

    # For demonstration, we'll classify only the first 24 records (one day's worth of hourly data).
    for i, record in enumerate(records[:24]):
        label = classify_record(client, record)
        enriched.append({**record, "conditions": label})

        # Print progress every 6 records (every 6 hours)
        if (i + 1) % 6 == 0:
            print(f"Processed {i + 1} records...")

    # Step 3: Load - save the enriched data back to Azure Blob Storage
    payload = json.dumps(enriched, indent=2).encode("utf-8")
    container.upload_blob(
        processed_blob_path,
        payload,
        overwrite=True,
    )

    print(f"Uploaded enriched data to {processed_blob_path}")

    downloaded = container.download_blob(processed_blob_path).readall()
    checked_records = json.loads(downloaded.decode("utf-8"))

    # Step 4: Spot-Check the results
    df = pd.DataFrame(checked_records)

    print("\nCondition counts:")
    print(df["conditions"].value_counts())

    print("\nFirst 5 rows:")
    print(df.head())

    # Step 5: Save Output
    output_dir = Path("assignments_10/outputs")
    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / "first_10_records.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(enriched[:10], f, indent=2)

    print(f"\nSaved first 10 records to {output_path}")


# Run the main function when this script is executed
if __name__ == "__main__":
    main()
