"""
Real data ingestion pipeline.

Data source: Dubai_Real_Estate_Dataset.csv
Source URL: https://raw.githubusercontent.com/Sukesh1985/Dubai-real-estate-Power-BI/main/Dubai_Real_Estate_Dataset.csv
Rows: ~2,000 real Dubai property listings

Embedding model: text-embedding-3-large (OpenAI, 3072-dim)
Reason: Best available semantic embedding accuracy via API, zero local GPU needed.
Ingests in batches of 50 to stay within OpenAI rate limits.
"""
import os
import sys
import csv
import requests
from dotenv import load_dotenv

load_dotenv()

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import text
from app.db.session import SessionLocal, engine, Base
from app.db.models import Property
from app.embeddings import get_embeddings_batch

DATASET_URL = "https://raw.githubusercontent.com/Sukesh1985/Dubai-real-estate-Power-BI/main/Dubai_Real_Estate_Dataset.csv"
BATCH_SIZE = 50  # OpenAI embedding API batch limit sweet spot


def build_embed_text(row: dict) -> str:
    """
    Construct a rich, descriptive text for semantic embedding.
    This text is what the vector search will compare against user queries.
    The richer and more natural this text, the more accurate the similarity search.
    """
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


def run_ingestion():
    # Step 1: Drop and recreate the table cleanly
    print("⚠️  Dropping old properties table and recreating with new schema...")
    with engine.connect() as conn:
        conn.execute(text("DROP TABLE IF EXISTS properties CASCADE"))
        conn.commit()

    # Recreate with the new schema (including Vector(3072))
    Base.metadata.create_all(bind=engine)
    print("✅ Schema created: properties table with vector(3072) for text-embedding-3-large")

    # Step 2: Download the real dataset
    print(f"\n📡 Downloading real Dubai property dataset from GitHub...")
    response = requests.get(DATASET_URL, timeout=30)
    response.raise_for_status()
    lines = response.text.splitlines()
    reader = csv.DictReader(lines)
    rows = list(reader)
    print(f"✅ Downloaded {len(rows)} real Dubai property records")

    # Step 3: Process in batches
    db = SessionLocal()
    total = len(rows)
    inserted = 0

    try:
        for batch_start in range(0, total, BATCH_SIZE):
            batch = rows[batch_start: batch_start + BATCH_SIZE]

            # Build embed texts for the whole batch
            embed_texts = [build_embed_text(row) for row in batch]

            # Get all embeddings in a single API call (efficient)
            print(f"  🔄 Embedding batch {batch_start // BATCH_SIZE + 1} ({len(batch)} records)...")
            embeddings = get_embeddings_batch(embed_texts)

            # Insert into DB
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

    except Exception as e:
        db.rollback()
        print(f"❌ Ingestion failed at batch {batch_start}: {e}")
        raise
    finally:
        db.close()

    print(f"\n🎉 Ingestion complete! {inserted} real Dubai properties stored with 3072-dim embeddings.")


if __name__ == "__main__":
    run_ingestion()
