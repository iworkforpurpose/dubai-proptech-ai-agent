from sqlalchemy import Column, Integer, String, Float
from pgvector.sqlalchemy import Vector
from app.db.session import Base

class Property(Base):
    __tablename__ = "properties"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, index=True)
    description = Column(String)
    location = Column(String, index=True)
    property_type = Column(String)
    bedrooms = Column(Integer)
    bathrooms = Column(Integer)
    price_aed = Column(Float)
    amenities = Column(String) # Stored as comma-separated string for simplicity
    status = Column(String, default="available")
    
    # 1536 is the dimension size for OpenAI's text-embedding-3-small model
    embedding = Column(Vector(1536)) 
