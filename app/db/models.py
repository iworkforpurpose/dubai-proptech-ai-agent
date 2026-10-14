from sqlalchemy import Column, Integer, String, Float, Text
from pgvector.sqlalchemy import Vector
from app.db.session import Base


class Property(Base):
    __tablename__ = "properties"

    id = Column(Integer, primary_key=True, index=True)
    property_id = Column(String, unique=True, index=True)   # e.g. DXB00001
    property_type = Column(String, index=True)              # Apartment, Villa, Penthouse, etc.
    location = Column(String, index=True)                   # Palm Jumeirah, Downtown Dubai, etc.
    bedrooms = Column(Integer)
    bathrooms = Column(Integer)
    size_sqft = Column(Float)
    price_aed = Column(Float)
    price_per_sqft = Column(Float)
    furnishing = Column(String)                             # Furnished / Unfurnished / Semi-Furnished
    year_built = Column(Integer)
    amenities = Column(Text)                                # Comma-separated amenities string
    view_type = Column(String)                              # Sea View, Burj Khalifa View, City View, etc.
    parking_spaces = Column(Integer)
    property_age = Column(Integer)

    # text-embedding-3-large produces 3072-dimensional vectors
    # This is the best OpenAI embedding model for semantic search
    embedding = Column(Vector(3072))
