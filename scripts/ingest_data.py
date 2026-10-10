import os
import sys
import json
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openai import OpenAI
from app.db.session import SessionLocal
from app.db.models import Property

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_embedding(text):
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

def ingest_data():
    if not os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY") == "your_openai_api_key_here":
        print("ERROR: Please add your OPENAI_API_KEY to the .env file.")
        return

    db = SessionLocal()
    
    with open("mock_dubai_properties.json", "r") as f:
        properties = json.load(f)
        
    print(f"Loaded {len(properties)} properties. Starting vector embedding and database insertion...")
    
    for i, prop in enumerate(properties):
        # We create a rich text representation to embed so the AI can search it semantically
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
    print("Ingestion complete! All properties are now stored as vectors.")

if __name__ == "__main__":
    ingest_data()
