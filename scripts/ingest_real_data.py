"""
Self-healing real Dubai property data ingestion pipeline using Google gemini-embedding-2.

Features:
- Idempotent: Skips already ingested properties via Property_ID index.
- Rate-limit aware: Batches requests with pacing to stay well within free tier limits.
- Automatic 429 retry: Backs off and resumes if throttled.
- Clean progress logging.
"""
import os
import sys
import csv
import time
import requests
from dotenv import load_dotenv

load_dotenv()

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import text
from app.db.session import SessionLocal, engine, Base
from app.db.models import Property
from app.embeddings import get_embeddings_batch

DATASET_URL = "https://raw.githubusercontent.com/Sukesh1985/Dubai-real-estate-Power-BI/main/Dubai_Real_Estate_Dataset.csv"
BATCH_SIZE = 50  # 50 records per batch in a single API request
DELAY_BETWEEN_BATCHES_SEC = 3.0  # Safe pacing to respect Google API rate limits



def build_embed_text(row: dict) -> str:
    """Construct semantic embedding text representation of real property."""
    return (
        f"{row['Property_Type']} for sale in {row['Location']}, Dubai. "
        f"{row['Bedrooms']} bedrooms, {row['Bathrooms']} bathrooms, "
        f"{row['Size_sqft']} sqft. "
        f"Price: {float(row['Price_AED']):,.0f} AED ({row['Price_per_sqft']} AED per sqft). "
        f"Furnishing: {row['Furnishing']}. "
        f"View: {row['View_Type']}. "
        f"Amenities: {row['Amenities']}. "
        f"Built in {row['Year_Built']}. Parking spaces: {row['Parking_Spaces']}."
    )


def run_ingestion(limit: int = None, reset_table: bool = False):
    # Step 1: Initialize schema if needed
    if reset_table:
        print("⚠️  Dropping old properties table and recreating with Vector(3072)...")
        with engine.connect() as conn:
            conn.execute(text("DROP TABLE IF EXISTS properties CASCADE"))
            conn.commit()
    Base.metadata.create_all(bind=engine)

    # Step 2: Download dataset
    print("\n📡 Fetching real Dubai properties dataset...")
    response = requests.get(DATASET_URL, timeout=30)
    response.raise_for_status()
    lines = response.text.splitlines()
    reader = csv.DictReader(lines)
    all_rows = list(reader)

    if limit:
        all_rows = all_rows[:limit]

    # Step 3: Find existing property IDs in DB to enable resume
    db = SessionLocal()
    try:
        existing_ids = set(
            row[0] for row in db.execute(text("SELECT property_id FROM properties")).fetchall()
        )
    finally:
        db.close()

    rows_to_ingest = [r for r in all_rows if r["Property_ID"] not in existing_ids]
    print(f"✅ Found {len(existing_ids)} existing properties in database.")
    print(f"📦 {len(rows_to_ingest)} new properties remaining to ingest (out of {len(all_rows)} total).")

    if not rows_to_ingest:
        print("🎉 Database is already fully populated with all requested properties!")
        return

    # Step 4: Batch embed and insert with rate limit protection
    total = len(rows_to_ingest)
    inserted = 0

    for batch_start in range(0, total, BATCH_SIZE):
        batch = rows_to_ingest[batch_start: batch_start + BATCH_SIZE]
        embed_texts = [build_embed_text(r) for r in batch]
        batch_num = (batch_start // BATCH_SIZE) + 1
        total_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE

        # Embed with automatic 429 retry
        embeddings = None
        for attempt in range(5):
            try:
                print(f"  🔄 Embedding batch {batch_num}/{total_batches} ({len(batch)} records)...")
                embeddings = get_embeddings_batch(embed_texts)
                break
            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                    print(f"  ⏳ Rate limit reached. Backing off for 60s before retrying batch {batch_num}...")
                    time.sleep(60)
                else:
                    print(f"  ❌ Non-rate-limit error: {e}")
                    raise

        if embeddings is None:
            print(f"❌ Failed to embed batch {batch_num} after retries. Aborting.")
            break

        # Insert batch into PostgreSQL
        db = SessionLocal()
        try:
            for row, embedding in zip(batch, embeddings):
                prop = Property(
                    property_id=row["Property_ID"],
                    property_type=row["Property_Type"],
                    location=row["Location"],
                    bedrooms=int(row["Bedrooms"]),
                    bathrooms=int(row["Bathrooms"]),
                    size_sqft=float(row["Size_sqft"]),
                    price_aed=float(row["Price_AED"]),
                    price_per_sqft=float(row["Price_per_sqft"]),
                    furnishing=row["Furnishing"],
                    year_built=int(row["Year_Built"]),
                    amenities=row["Amenities"],
                    view_type=row["View_Type"],
                    parking_spaces=int(row["Parking_Spaces"]),
                    property_age=int(row["Property_Age"]),
                    embedding=embedding,
                )
                db.add(prop)
            db.commit()
            inserted += len(batch)
            print(f"  ✅ Inserted {len(batch)} items. Total in DB: {len(existing_ids) + inserted}")
        except Exception as e:
            db.rollback()
            print(f"  ❌ DB insertion error: {e}")
            raise
        finally:
            db.close()

        # Respect free-tier request rate
        time.sleep(DELAY_BETWEEN_BATCHES_SEC)

    print(f"\n🎉 Ingestion cycle complete! Current property count in DB: {len(existing_ids) + inserted}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Limit number of rows to ingest")
    parser.add_argument("--reset", action="store_true", help="Reset DB table before ingesting")
    args = parser.parse_args()
    run_ingestion(limit=args.limit, reset_table=args.reset)
