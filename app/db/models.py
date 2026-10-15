from datetime import datetime, timezone
from sqlalchemy import Column, Integer, String, Float, Text, DateTime
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

    # Google gemini-embedding-001 / gemini-embedding-2 output 3072-dimensional vectors
    embedding = Column(Vector(3072))


class Lead(Base):
    __tablename__ = "leads"

    id = Column(Integer, primary_key=True, index=True)
    full_name = Column(String, nullable=False)
    contact_info = Column(String, nullable=False)
    preferred_location = Column(String, nullable=True)
    budget_max_aed = Column(Float, nullable=True)
    property_id = Column(String, nullable=True)
    notes = Column(Text, nullable=True)
    status = Column(String, default="NEW_INQUIRY")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


