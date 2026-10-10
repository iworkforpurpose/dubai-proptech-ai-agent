import os
import sys
import json
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.db.session import SessionLocal
from app.db.models import Property

# Load high-accuracy local embedding model
print("Loading state-of-the-art local embedding model (BAAI/bge-large-en-v1.5)...")
model = SentenceTransformer('BAAI/bge-large-en-v1.5')

def get_embedding(text):
    # Returns a 384-dimensional numpy array, converted to list
    return model.encode(text).tolist()

def ingest_data():
    db = SessionLocal()
    
    with open("mock_dubai_properties.json", "r") as f:
        properties = json.load(f)
        
    print(f"Loaded {len(properties)} properties. Starting local vector embedding and database insertion...")
    
    for i, prop in enumerate(properties):
        embed_text = f"Title: {prop['title']}. Location: {prop['location']}. Type: {prop['property_type']}. " \
                     f"Price: {prop['price_aed']} AED. Bedrooms: {prop['bedrooms']}. Bathrooms: {prop['bathrooms']}. " \
                     f"Amenities: {', '.join(prop['amenities'])}. Description: {prop['description']}"
                     
        vector = get_embedding(embed_text)
        
        db_prop = Property(
            title=prop["title"],
            description=prop["description"],
            location=prop["location"],
            property_type=prop["property_type"],
            bedrooms=prop["bedrooms"],
            bathrooms=prop["bathrooms"],
            price_aed=prop["price_aed"],
            amenities=", ".join(prop["amenities"]),
            status=prop["status"],
            embedding=vector
        )
        db.add(db_prop)
        
        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1}/{len(properties)} properties...")
            
    db.commit()
    db.close()
    print("Ingestion complete! All properties are now stored as vectors using local embeddings.")

if __name__ == "__main__":
    ingest_data()
