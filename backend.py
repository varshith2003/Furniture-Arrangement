from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict
import numpy as np
import tensorflow as tf
import joblib

# Load trained scalers
X_scaler = joblib.load("X_scaler.pkl")  # Feature scaler
Y_scaler = joblib.load("Y_scaler.pkl")  # Label scaler

model = tf.keras.models.load_model(
    "mlp_furniture_arrangement.h5", compile=False)
# Ensure loss function is correct
model.compile(optimizer="adam", loss="mse")

app = FastAPI()

# Define request format


class FurnitureRequest(BaseModel):
    room_type: int
    room_width: float
    room_height: float
    # Each obstacle: {"type": int, "x": float, "y": float, "width": float, "height": float}
    obstacles: List[Dict[str, float]]
    # Each furniture: {"type": int, "width": float, "height": float}
    furniture_list: List[Dict[str, float]]


@app.post("/predict")
async def predict_layout(data: FurnitureRequest):
    # Step 1: Prepare Room Data (3 values)
    features = [data.room_type, data.room_width, data.room_height]

    # Step 2: Prepare Obstacle Data (Max 3 obstacles, each with 4 values)
    obstacle_features = []
    for obstacle in data.obstacles[:3]:  # Only take up to 3 obstacles
        obstacle_features.extend(
            [obstacle["x"], obstacle["y"], obstacle["width"], obstacle["height"]])

    while len(obstacle_features) < 12:  # Pad with zeros if fewer than 3 obstacles
        obstacle_features.extend([0, 0, 0, 0])

    # Step 3: Prepare Furniture Data (Max 7 furniture items, each with 3 values)
    furniture_features = []
    for item in data.furniture_list[:7]:  # Only take up to 7 furniture items
        furniture_features.extend(
            [item["type"], item["width"], item["height"]])

    while len(furniture_features) < 21:  # Pad with zeros if fewer than 7 items
        furniture_features.extend([0, 0, 0])

    # Step 4: Combine All Features into a 36-Dimensional Vector
    features.extend(obstacle_features)
    features.extend(furniture_features)

    # **Debugging Step**: Print the length of features
    # Should print **36**, not 37
    print("Feature Vector Length:", len(features))

    # Convert to NumPy array and reshape
    features = np.array(features, dtype=np.float32).reshape(1, -1)

    # Ensure Correct Shape Before Scaling
    assert features.shape[
        1] == 36, f"Feature vector should be 36, but got {features.shape[1]}"

    # Step 5: Normalize Input Data
    features = X_scaler.transform(features)  # Apply MinMax Scaling

    # Step 6: Make Prediction
    scaled_predictions = model.predict(features)

    print(scaled_predictions)

    # Step 7: Scale Back Predicted Values
    predictions = Y_scaler.inverse_transform(scaled_predictions)

    num_items = min(7, len(data.furniture_list))
    transformed_predictions = []
    for i in range(2 * num_items):  # Each item has an (x, y) coordinate
        if i % 2 == 0:  # X-coordinate
            transformed_predictions.append(predictions[0][i] * data.room_width)
        else:  # Y-coordinate
            transformed_predictions.append(predictions[0][i] * data.room_height)

    # Step 8: Mask Unused Positions with (0, 0)
    # Ensure only the first 2 * num_items coordinates are kept, rest set to 0
    max_positions = 14  # Maximum allowed (x, y) pairs → 7 furniture items × 2
    num_valid = 2* num_items  # Number of valid (x, y) values

    # Keep only the required positions and set the rest to 0
    transformed_predictions = transformed_predictions[:num_valid]
    # Ensure JSON-serializable output
    return {"predicted_positions": transformed_predictions}
