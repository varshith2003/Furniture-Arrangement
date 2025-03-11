import random
import streamlit as st
import requests
import matplotlib.pyplot as plt
import numpy as np


room_furniture = {
    "Bedroom": ["Bed", "Wardrobe", "Desk", "Bookshelf", "Storage Cabinet"],
    "Living Room": ["Sofa", "Coffee Table", "TV Stand", "Armchair", "Console Table"],
    "Dining Room": ["Dining Table", "Dining Chairs", "Sideboard", "Cabinets"],
    "Office": ["Desk", "Office Chair", "Bookshelf", "Filing Cabinet"],
    "Kitchen": ["Fridge", "Oven", "Cabinets", "Sink", "Microwave"],
    "Bathroom": ["Sink", "Toilet", "Bathtub", "Cabinets", "Mirror"]
}

furniture_sizes = {
    "Bed": (2.0, 1.5),
    "Single Bed": (2.0, 1.0),
    "Double Bed": (2.5, 1.5),
    "Wardrobe": (1.2, 0.6),
    "Side Table": (0.5, 0.5),
    "Desk": (1.5, 0.8),
    "Bookshelf": (1.0, 0.5),
    "Storage Cabinet": (1.5, 0.6),
    "Sofa": (2.2, 1.0),
    "Coffee Table": (1.2, 0.6),
    "TV Stand": (1.8, 0.5),
    "Armchair": (0.8, 0.8),
    "Console Table": (1.2, 0.4),
    "Dining Table": (2.0, 1.2),
    "Dining Chairs": (0.5, 0.5),
    "Sideboard": (1.5, 0.6),
    "Cabinets": (1.8, 0.5),
    "Office Chair": (0.6, 0.6),
    "Filing Cabinet": (1.2, 0.5),
    "Fridge": (0.8, 0.7),
    "Oven": (0.9, 0.6),
    "Sink": (1.0, 0.5),
    "Microwave": (0.6, 0.4),
    "Toilet": (0.7, 0.5),
    "Bathtub": (1.8, 0.8),
    "Mirror": (0.8, 0.4)
}


room_mapping = {
    0: "Bedroom",
    1: "Dining Room",
    2: "Living Room",
    3: "Office",
    4: "Kitchen",
    5: "Bathroom"
}


furniture_mapping = {
    "Armchair": 0,
    "Bathtub": 1,
    "Dining Chairs": 2,
    "Dining Table": 3,
    "Filing Cabinet": 4,
    "Fridge": 5,
    "Microwave": 6,
    "Mirror": 7,
    "Office Chair": 8,
    "Oven": 9,
    "Sideboard": 10,
    "Sink": 11,
    "Bed": 12,
    "Sofa": 13,
    "Storage Cabinet": 14,
    "TV Stand": 15,
    "Toilet": 16,
    "Wardrobe": 17,
    "Bookshelf": 18,
    "Cabinets": 19,
    "Chair": 20,
    "Coffee Table": 21,
    "Console Table": 22,
    "Couch": 23,
    "Desk": 24
}


# Streamlit UI
st.title("Furniture Arrangement AI")
type_of_room = st.selectbox(
    "Select Room Type", list(room_furniture.keys()), index=0)

default_furniture = room_furniture[type_of_room]
selected_furniture = st.multiselect(
    "Select Furniture", default_furniture, default=default_furniture)

# Room dimensions
room_width = st.number_input("Room Width (meters)", min_value=2.0, value=4.5)
room_height = st.number_input("Room Height (meters)", min_value=2.0, value=3.0)

# Obstacles
obstacle_count = st.number_input(
    "Number of Obstacles", min_value=0, value=0, step=1)
obstacles = []
for i in range(obstacle_count):
    x = st.number_input(f"Obstacle {i+1} X (meters)", min_value=0.0, value=1.0)
    y = st.number_input(f"Obstacle {i+1} Y (meters)", min_value=0.0, value=1.0)
    w = st.number_input(
        f"Obstacle {i+1} Width (meters)", min_value=0.1, value=0.5)
    h = st.number_input(
        f"Obstacle {i+1} Height (meters)", min_value=0.1, value=0.5)
    obstacles.append({"x": x, "y": y, "width": w, "height": h})

furniture_list = [
    {"type": furniture_mapping[f], "width": furniture_sizes[f]
     [0], "height": furniture_sizes[f][1]}
    for f in selected_furniture if f in furniture_mapping
]


def get_furniture_id(name):
    for key, value in furniture_mapping.items():
        if value == name:
            return key
    return None


def get_room_id(name):
    for key, value in room_mapping.items():
        if value == name:
            return key
    return None


request_data = {
    "room_type": get_room_id(type_of_room),
    "room_width": room_width,
    "room_height": room_height,
    "obstacles": obstacles,
    "furniture_list": furniture_list
}


def optimize_furniture_positions(room_width, room_height, furniture, positions):
    optimized_positions = []
    occupied_spaces = []

    def is_overlapping(x, y, width, height):
        """Check if the new furniture overlaps with any existing furniture."""
        for ox, oy, ow, oh in occupied_spaces:
            if (x < ox + ow and x + width > ox and y < oy + oh and y + height > oy):
                return True
        return False

    for i, item in enumerate(furniture):
        width, height = item["width"], item["height"]
        max_attempts = 100  # Limit the number of attempts to find a valid placement
        attempt = 0

        while attempt < max_attempts:
            # Randomize positions within room boundaries
            x = random.uniform(0, room_width - width)
            y = random.uniform(0, room_height - height)

            if not is_overlapping(x, y, width, height):
                optimized_positions.append((x, y))
                occupied_spaces.append((x, y, width, height))
                break  # Exit loop once a valid position is found

            attempt += 1

        # If no valid position found after max attempts, place it at (0,0) (fallback)
        if attempt == max_attempts:
            optimized_positions.append((0, 0))
            occupied_spaces.append((0, 0, width, height))

    return optimized_positions


def plot_room_layout(room_width, room_height, furniture, positions, obstacles):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, room_width)
    ax.set_ylim(0, room_height)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(type_of_room)

    # Draw room boundary
    ax.add_patch(plt.Rectangle((0, 0), room_width, room_height,
                 fill=None, edgecolor='black', linewidth=2))

    # Draw obstacles
    for obs in obstacles:
        ox, oy, ow, oh = obs["x"], obs["y"], obs["width"], obs["height"]
        ax.add_patch(plt.Rectangle((ox, oy), ow, oh,
                     fill=True, color='red', alpha=0.7))
        ax.text(ox + ow / 2, oy + oh / 2, "Obstacle", ha='center',
                va='center', fontsize=8, color='white')

    # Ensure the positions are in (x, y) pairs
    if len(positions) % 2 != 0:
        st.error("Invalid position data received from backend")
        return

    optimized_positions = optimize_furniture_positions(
        room_width, room_height, furniture, positions)

    if len(optimized_positions) != len(furniture):
        st.error("Mismatch between predicted positions and furniture items")
        return

    for item, (x, y) in zip(furniture, optimized_positions):
        width, height = item["width"], item["height"]
        name = get_furniture_id(item['type'])
        ax.add_patch(plt.Rectangle((x, y), width, height,
                     fill=True, color=np.random.rand(3,)))
        ax.text(x + width / 2, y + height / 2, name, ha='center',
                va='center', fontsize=8, color='white')

    st.pyplot(fig)



# Call Backend API and plot layout
if st.button("Generate Layout Visualization"):
    response = requests.post(
        "http://127.0.0.1:8000/predict", json=request_data)
    if response.status_code == 200:
        layout = response.json()
        plot_room_layout(room_width, room_height, furniture_list, layout["predicted_positions"], obstacles)
    else:
        st.error("Failed to generate layout. Please check the backend service.")
