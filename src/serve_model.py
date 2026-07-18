import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# Redefine the architecture so Keras can map the saved weights
class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

input_dim = 4
latent_dim = 2

# Rebuild Encoder
encoder_inputs = keras.Input(shape=(input_dim,))
x = layers.Dense(16, activation="relu")(encoder_inputs)
x = layers.Dense(8, activation="relu")(x)
z_mean = layers.Dense(latent_dim, name="z_mean")(x)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
z = Sampling()([z_mean, z_log_var])
encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

# Rebuild Decoder
latent_inputs = keras.Input(shape=(latent_dim,))
x = layers.Dense(8, activation="relu")(latent_inputs)
x = layers.Dense(16, activation="relu")(x)
decoder_outputs = layers.Dense(input_dim, activation="linear")(x)
decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")

# Rebuild Full VAE
class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def call(self, inputs):
        z_mean, _, _ = self.encoder(inputs)
        return self.decoder(z_mean)

vae = VAE(encoder, decoder)
# Initialize weights by calling the model once with dummy data
vae(tf.zeros((1, input_dim))) 

# Load the saved weights
print("Loading model weights...")
vae.load_weights('data/processed/vae_model/vae_weights.weights.h5')

# Load the preprocessing scaler
print("Loading scaler...")
scaler = joblib.load('data/processed/scaler.pkl')

# Set the anomaly threshold
ANOMALY_THRESHOLD = 0.4273712476419824

# --- 2. FastAPI Application Setup ---
app = FastAPI(title="Web Attack VAE Anomaly Detector", version="1.0")

# Define the expected input payload schema
class WebLog(BaseModel):
    bytes_in: float
    bytes_out: float
    response_code: float
    dst_port: float

@app.post("/predict")
def predict_anomaly(log: WebLog):
    try:
        # 1. Extract and format features
        features = np.array([[log.bytes_in, log.bytes_out, log.response_code, log.dst_port]])
        
        # 2. Scale features using the fitted normal scaler
        scaled_features = scaler.transform(features)
        
        # 3. Reconstruct via VAE
        reconstruction = vae.predict(scaled_features, verbose=0)
        
        # 4. Calculate MSE for this specific log
        mse = np.mean(np.square(scaled_features - reconstruction), axis=1)[0]
        
        # 5. Determine if it is an attack
        is_anomaly = bool(mse > ANOMALY_THRESHOLD)
        
        return {
            "is_anomaly": is_anomaly,
            "reconstruction_error": float(mse),
            "threshold": ANOMALY_THRESHOLD,
            "status": "Attack Detected!" if is_anomaly else "Normal Traffic"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    print("Starting API Server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)