import logging
import os
import pandas as pd
from pymongo import MongoClient
from tqdm import tqdm
from urllib.parse import quote_plus

# ── Logging setup ──────────────────────────────────────
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('logs/ingestion.log'),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)


def get_category(aqi):
    """Map AQI value to EPA health category."""
    if pd.isna(aqi):
        return "Unknown"
    aqi = int(aqi)
    if aqi <= 50:   return "Good"
    if aqi <= 100:  return "Moderate"
    if aqi <= 150:  return "Unhealthy for Sensitive Groups"
    if aqi <= 200:  return "Unhealthy"
    if aqi <= 300:  return "Very Unhealthy"
    return "Hazardous"


USERNAME  = quote_plus('')
PASSWORD  = quote_plus('')
MONGO_URI = f'mongodb+srv://{USERNAME}:{PASSWORD}@nosqlfinalproject.vkq6zwb.mongodb.net/?appName=NoSQLFinalProject'
CSV_FILE  = "daily_88502_2023.csv"


log.info('=' * 60)
log.info('DS 4320 Project 2 — PM2.5 Ingest (88502)')
log.info('=' * 60)

# Connect
try:
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=15_000)
    client.admin.command('ping')
    log.info('Connected to MongoDB Atlas.')
except Exception as exc:
    log.error('MongoDB connection failed: %s', exc)
    raise

col = client["air_quality_db"]["pm25_daily"]

# Read CSV
log.info('Reading CSV: %s', CSV_FILE)
try:
    df = pd.read_csv(CSV_FILE, low_memory=False)
    log.info('Loaded %d raw rows.', len(df))
except Exception as exc:
    log.error('Failed to read CSV: %s', exc)
    raise

# Clean data
df["Date Local"] = pd.to_datetime(df["Date Local"])
df["AQI"] = pd.to_numeric(df["AQI"], errors="coerce")
df["Arithmetic Mean"] = pd.to_numeric(df["Arithmetic Mean"], errors="coerce")

before = len(df)
df = df[(df["Arithmetic Mean"] >= 0) & (df["AQI"] <= 500)].dropna(subset=["Arithmetic Mean"])
log.info('Cleaned outliers: %d -> %d rows (removed %d).',
         before, len(df), before - len(df))

# Upload in batches
inserted = 0
for i in tqdm(range(0, len(df), 500), desc="Uploading"):
    batch = df.iloc[i: i + 500]
    docs = []
    for _, row in batch.iterrows():
        docs.append({
            "date":        row["Date Local"].to_pydatetime(),
            "year":        int(row["Date Local"].year),
            "month":       int(row["Date Local"].month),
            "day_of_week": row["Date Local"].strftime("%A"),
            "location": {
                "city":      row.get("City Name"),
                "county":    row.get("County Name"),
                "state":     row.get("State Name"),
                "cbsa_name": row.get("CBSA Name"),
                "latitude":  float(row["Latitude"])  if pd.notna(row.get("Latitude"))  else None,
                "longitude": float(row["Longitude"]) if pd.notna(row.get("Longitude")) else None,
            },
            "air_quality": {
                "pm25_mean":    round(float(row["Arithmetic Mean"]), 2),
                "pm25_max":     float(row["1st Max Value"]) if pd.notna(row.get("1st Max Value")) else None,
                "aqi":          int(row["AQI"])             if pd.notna(row.get("AQI"))           else None,
                "aqi_category": get_category(row.get("AQI")),
            },
            "provenance": {
                "source":    "EPA AQS",
                "parameter": "PM2.5 Speciation Mass (88502)",
            },
        })
    try:
        col.insert_many(docs)
        inserted += len(docs)
    except Exception as exc:
        log.error('Batch insert failed at row %d: %s', i, exc)

log.info('Upload complete. Inserted %d documents.', inserted)
log.info('Total in collection: %d', col.count_documents({}))
client.close()
log.info('Done.')