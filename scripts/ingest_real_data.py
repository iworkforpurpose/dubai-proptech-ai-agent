"""
Real Dubai property data ingestion pipeline using Google Gemini text-embedding-004.

Data source: Dubai_Real_Estate_Dataset.csv (~2,000 real Dubai listings)
Source: https://raw.githubusercontent.com/Sukesh1985/Dubai-real-estate-Power-BI/main/Dubai_Real_Estate_Dataset.csv
Embedding model: text-embedding-004 (768 dimensions)
Database: PostgreSQL + pgvector
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
BATCH_SIZE = 30  # Sized for rate-limit safety and fast processing


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


def run_ingestion(limit: int = None):
    # Step 1: Re-create schema with Vector(768)
    print("⚠️  Dropping old properties table and recreating with Vector(768)...")
    with engine.connect() as conn:
        conn.execute(text("DROP TABLE IF EXISTS properties CASCADE"))
        conn.commit()

    Base.metadata.create_all(bind=engine)
    print("✅ Schema created: properties table with Vector(768) for text-embedding-004")

    # Step 2: Download real dataset
    print("\n📡 Fetching real Dubai properties dataset...")
    response = requests.get(DATASET_URL, timeout=30)
    response.raise_for_status()
    lines = response.text.splitlines()
    reader = csv.DictReader(lines)
    rows = list(reader)

    if limit:
        rows = rows[:limit]

    total = len(rows)
    print(f"✅ Loaded {total} real Dubai listings for ingestion.")

    # Step 3: Embed and insert in batches
    db = SessionLocal()
    inserted = 0

    try:
        for batch_start in range(0, total, BATCH_SIZE):
            batch = rows[batch_start: batch_start + BATCH_SIZE]
            embed_texts = [build_embed_text(r) for r in batch]

            batch_num = (batch_start // BATCH_SIZE) + 1
            total_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE
            print(f"  🔄 Embedding batch {batch_num}/{total_batches} ({len(batch)} records)...")

            # Call text-embedding-004
            embeddings = get_embeddings_batch(embed_texts)

            # Insert batch into PostgreSQL
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
            print(f"  ✅ Inserted {inserted}/{total} properties")

            # Brief pause between batches to prevent free-tier throttling
            time.sleep(0.5)

    except Exception as e:
        db.rollback()
        print(f"❌ Ingestion stopped at batch starting at index {batch_start}: {e}")
        raise
    finally:
        db.close()

    print(f"\n🎉 Ingestion complete! Successfully ingested {inserted} real Dubai properties into PostgreSQL with 768-dim embeddings.")


if __name__ == "__main__":
    run_ingestion()
