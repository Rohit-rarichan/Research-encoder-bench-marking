IGNORE_INDEX = 255

# Keep this small at first
CLASS_NAMES = [
    "vehicle",
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
    category_name = category_name.lower()

    if "vehicle" in category_name or "car" in category_name or "truck" in category_name or "bus" in category_name:
        return CLASS_NAME_TO_ID["vehicle"]

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