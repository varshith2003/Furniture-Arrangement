import numpy as np

# Room ID Mapping
ROOM_MAPPING = {
    "Bedroom": 0, "Dining Room": 1, "Living Room": 2, "Office": 3, "Kitchen": 4, "Bathroom": 5
}

# Furniture ID Mapping
FURNITURE_MAPPING = {
    "Armchair": 0, "Bathtub": 1, "Dining Chairs": 2, "Dining Table": 3, "Filing Cabinet": 4,
    "Fridge": 5, "Microwave": 6, "Mirror": 7, "Office Chair": 8, "Oven": 9, "Sideboard": 10,
    "Sink": 11, "Bed": 12, "Sofa": 13, "Storage Cabinet": 14, "TV Stand": 15, "Toilet": 16,
    "Wardrobe": 17, "Bookshelf": 18, "Cabinets": 19, "Chair": 20, "Coffee Table": 21,
    "Console Table": 22, "Couch": 23, "Desk": 24
}

# Furniture options per room type
FURNITURE_OPTIONS = {
    "Bedroom": ["Bed", "Wardrobe", "Desk", "Bookshelf", "Storage Cabinet"],
    "Living Room": ["Sofa", "Coffee Table", "TV Stand", "Armchair", "Console Table"],
    "Dining Room": ["Dining Table", "Dining Chairs", "Sideboard", "Cabinets"],
    "Office": ["Desk", "Office Chair", "Bookshelf", "Filing Cabinet"],
    "Kitchen": ["Fridge", "Oven", "Cabinets", "Sink", "Microwave"],
    "Bathroom": ["Sink", "Toilet", "Bathtub", "Cabinets", "Mirror"]
}

# Dataset parameters
num_samples = 3300
num_features = 50

# Room size constraints
ROOM_MIN_SIZE = (5, 5)
ROOM_MAX_SIZE = (12, 12)

# Furniture size constraints (width, height range)
FURNITURE_SIZE_RANGE = (1, 3)

# Obstacles size constraints (width, height range)
OBSTACLE_SIZE_RANGE = (1, 4)

# Generate dataset
dataset = np.zeros((num_samples, num_features))

for i in range(num_samples):
    # Select random room type and dimensions
    room_type = np.random.choice(list(ROOM_MAPPING.keys()))
    room_id = ROOM_MAPPING[room_type]
    room_width = np.random.randint(ROOM_MIN_SIZE[0], ROOM_MAX_SIZE[0] + 1)
    room_height = np.random.randint(ROOM_MIN_SIZE[1], ROOM_MAX_SIZE[1] + 1)

    # Generate up to 3 obstacles (each with x, y, width, height), or [0,0,0,0] if none exist
    obstacles = []
    for _ in range(3):
        if np.random.rand() < 0.7:  # 70% chance to place an obstacle
            ob_w = np.random.randint(
                OBSTACLE_SIZE_RANGE[0], OBSTACLE_SIZE_RANGE[1] + 1)
            ob_h = np.random.randint(
                OBSTACLE_SIZE_RANGE[0], OBSTACLE_SIZE_RANGE[1] + 1)
            ob_x = np.random.randint(0, room_width - ob_w)
            ob_y = np.random.randint(0, room_height - ob_h)
            obstacles.extend([ob_x, ob_y, ob_w, ob_h])
        else:
            obstacles.extend([0, 0, 0, 0])  # No obstacle

    # Select 7 random furniture items for the room
    # Select a random number of furniture items (up to available count)
    # Ensure valid selection
    max_furniture = min(7, len(FURNITURE_OPTIONS[room_type]))
    furniture_list = np.random.choice(
        FURNITURE_OPTIONS[room_type], max_furniture, replace=False)

    furniture_data = []

    # Track placed furniture positions to avoid overlap
    placed_positions = []

    for furniture in furniture_list:
        f_width = np.random.randint(
            FURNITURE_SIZE_RANGE[0], FURNITURE_SIZE_RANGE[1] + 1)
        f_height = np.random.randint(
            FURNITURE_SIZE_RANGE[0], FURNITURE_SIZE_RANGE[1] + 1)

        # Find a valid placement (not overlapping obstacles or other furniture)
        max_attempts = 1000
        attempts = 0
        while attempts < max_attempts:
            f_x = np.random.randint(0, room_width - f_width)
            f_y = np.random.randint(0, room_height - f_height)
            valid = True

            # Check obstacle overlap
            for j in range(0, len(obstacles), 4):
                ob_x, ob_y, ob_w, ob_h = obstacles[j:j+4]
                if (f_x < ob_x + ob_w and f_x + f_width > ob_x and
                        f_y < ob_y + ob_h and f_y + f_height > ob_y):
                    valid = False
                    break

            # Check furniture overlap
            for placed in placed_positions:
                p_x, p_y, p_w, p_h = placed
                if (f_x < p_x + p_w and f_x + f_width > p_x and
                        f_y < p_y + p_h and f_y + f_height > p_y):
                    valid = False
                    break

            if valid:
                placed_positions.append((f_x, f_y, f_width, f_height))
                break

            attempts += 1

        # If placement fails, mark it as -1, -1 (unplaced furniture)
        if attempts == max_attempts:
            f_x, f_y = 0, 0

        # Store furniture type and dimensions
        furniture_data.extend(
            [FURNITURE_MAPPING[furniture], f_width, f_height])

    # Ensure exactly 7 furniture items, padding with [0,0,0] if needed
    while len(furniture_data) < 21:
        furniture_data.extend([0, 0, 0])

    # Ensure exactly 7 positions (x, y), padding with [0,0] if needed
    while len(placed_positions) < 7:
        placed_positions.append([0, 0])

    # Flatten placed positions and take only first 2 values per furniture
    flat_positions = [pos for p in placed_positions for pos in p[:2]]

    # Ensure the dataset row has exactly 50 elements
    dataset[i] = [room_id, room_width, room_height] + obstacles + \
        furniture_data[:21] + flat_positions[:14]
    
np.save("normalized_furniture_dataset", dataset)
