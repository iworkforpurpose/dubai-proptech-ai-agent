import json
import random

LOCATIONS = ["Downtown Dubai", "Dubai Marina", "Palm Jumeirah", "Jumeirah Village Circle (JVC)", "Business Bay"]
PROPERTY_TYPES = ["Apartment", "Villa", "Townhouse", "Penthouse"]
AMENITIES_LIST = ["Pool", "Gym", "Balcony", "Maid Room", "Sea View", "Burj Khalifa View", "Covered Parking", "Security"]

def generate_mock_properties(num=500):
    properties = []
    for i in range(1, num + 1):
        loc = random.choice(LOCATIONS)
        ptype = random.choice(PROPERTY_TYPES)
        bedrooms = random.randint(1, 5) if ptype != "Villa" else random.randint(3, 7)
        bathrooms = bedrooms + random.randint(0, 2)
        
        # Approximate pricing in AED
        base_price = bedrooms * 500000
        if loc in ["Downtown Dubai", "Palm Jumeirah"]:
            base_price = int(base_price * 1.5)
            
        price = base_price + random.randint(10000, 200000)
        
        # Amenities
        num_amenities = random.randint(2, 6)
        amenities = random.sample(AMENITIES_LIST, num_amenities)
        
        description = f"Beautiful {bedrooms} BR {ptype} in {loc}. Features include {', '.join(amenities)}."
        
        properties.append({
            "id": i,
            "title": f"Luxurious {bedrooms} BR {ptype} in {loc}",
            "description": description,
            "location": loc,
            "property_type": ptype,
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "price_aed": price,
            "amenities": amenities,
            "status": "available"
        })
        
    return properties

if __name__ == "__main__":
    data = generate_mock_properties(500)
    with open("mock_dubai_properties.json", "w") as f:
        json.dump(data, f, indent=2)
    print("Generated 500 mock properties to mock_dubai_properties.json")
