import joblib
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Load dataset
X = np.load("normalized_furniture_dataset.npy")
Y = X[:, -14:]  # Assuming last 14 columns are targets (furniture positions)
X = X[:, :-14]  # Features (first 36 columns)

# Split dataset: 80% Train, 10% Validation, 10% Test
X_train, X_temp, Y_train, Y_temp = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# Initialize different scalers
X_scaler = MinMaxScaler()
Y_scaler = MinMaxScaler()

# Fit and transform on train data (features & labels separately)
X_train = X_scaler.fit_transform(X_train)
Y_train = Y_scaler.fit_transform(Y_train)

# Transform (ONLY transform, NOT fit) test data
X_temp = X_scaler.transform(X_temp)
Y_temp = Y_scaler.transform(Y_temp)


# Save feature scaler
joblib.dump(X_scaler, "X_scaler.pkl")

# Save label scaler
joblib.dump(Y_scaler, "Y_scaler.pkl")


X_val, X_test, Y_val, Y_test = train_test_split(
    X_temp, Y_temp, test_size=0.5, random_state=42
)


def build_mlp_model():
    model = keras.Sequential([
        layers.Input(shape=(36,)),  # Input layer with 36 neurons
        # Hidden Layer 1 (16 neurons, ReLU)
        layers.Dense(16, activation='relu'),
        # Hidden Layer 2 (24 neurons, ReLU)
        layers.Dense(24, activation='relu'),
        # Output Layer (14 neurons, Linear)
        layers.Dense(14, activation='linear')
    ])
    return model


# Build and compile the model
model = build_mlp_model()
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Early stopping to prevent overfitting
early_stopping = EarlyStopping(
    monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
history = model.fit(
    X_train, Y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_val, Y_val),
    callbacks=[early_stopping]
)

# Evaluate the model on test data
test_loss, test_mae = model.evaluate(X_test, Y_test)
print(f"Test Loss (MSE): {test_loss:.4f}, Test MAE: {test_mae:.4f}")

# Save the trained model
model.save('mlp_furniture_arrangement.h5')
print("Model training complete. Saved as 'mlp_furniture_arrangement.h5'")
