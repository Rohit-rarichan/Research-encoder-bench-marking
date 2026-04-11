IGNORE_INDEX = 255

# Specific vehicle classes for accurate classification
CLASS_NAMES = [
    "car",
    "truck",
    "bus",
    "bicycle",
    "motorcycle",
    "pedestrian",
    "driveable_surface",
    "other_flat",
    "terrain",
    "manmade",
    "vegetation",
]

NUM_CLASSES = len(CLASS_NAMES)

CLASS_NAME_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}


def map_object_category(category_name: str):
    """Map nuimages category names to class IDs.
    
    Ensures accurate classification:
    - vehicle.car -> car class
    - vehicle.truck -> truck class  
    - vehicle.bus.* -> bus class
    - vehicle.bicycle -> bicycle class
    - vehicle.motorcycle -> motorcycle class
    - pedestrian* -> pedestrian class
    """
    category_name = category_name.lower()

    # Accurate vehicle classification
    if "vehicle.car" in category_name:
        return CLASS_NAME_TO_ID["car"]
    
    if "vehicle.truck" in category_name:
        return CLASS_NAME_TO_ID["truck"]
    
    if "vehicle.bus" in category_name:
        return CLASS_NAME_TO_ID["bus"]
    
    if "vehicle.bicycle" in category_name:
        return CLASS_NAME_TO_ID["bicycle"]
    
    if "vehicle.motorcycle" in category_name:
        return CLASS_NAME_TO_ID["motorcycle"]

    if "pedestrian" in category_name:
        return CLASS_NAME_TO_ID["pedestrian"]

    return None


def map_surface_category(category_name: str):
    category_name = category_name.lower()

    if "driveable_surface" in category_name or "driveable surface" in category_name:
        return CLASS_NAME_TO_ID["driveable_surface"]
    if "other_flat" in category_name or "other flat" in category_name:
        return CLASS_NAME_TO_ID["other_flat"]
    if "terrain" in category_name:
        return CLASS_NAME_TO_ID["terrain"]
    if "manmade" in category_name:
        return CLASS_NAME_TO_ID["manmade"]
    if "vegetation" in category_name:
        return CLASS_NAME_TO_ID["vegetation"]

    return None