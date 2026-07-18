import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import mlflow
import mlflow.tensorflow

# --- 1. Custom VAE Components ---

class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

class VAE(keras.Model):
    """Variational Autoencoder Custom Model."""
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        # Keras 3 passes data as a tuple if validation_data is provided
        if isinstance(data, tuple):
            data = data[0]
            
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            
            # Reconstruction Loss - Pure TF math to avoid Keras API versioning issues
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(tf.square(data - reconstruction), axis=1)
            )
            
            # KL Divergence Loss
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            
            total_loss = reconstruction_loss + kl_loss
            
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def test_step(self, data):
        if isinstance(data, tuple):
            data = data[0]

        z_mean, z_log_var, z = self.encoder(data)
        reconstruction = self.decoder(z)
        
        # Reconstruction Loss - Pure TF math
        reconstruction_loss = tf.reduce_mean(
            tf.reduce_sum(tf.square(data - reconstruction), axis=1)
        )
        
        kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
        total_loss = reconstruction_loss + kl_loss
        
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def call(self, inputs):
        # Allow the model to be called natively for inference
        z_mean, _, _ = self.encoder(inputs)
        return self.decoder(z_mean)


# --- 2. Data Loading & Preprocessing ---

def load_and_preprocess_data():
    print("Loading data...")
    normal_df = pd.read_csv('data/synthetic/normal_traffic.csv')
    attack_df = pd.read_csv('data/raw/CloudWatch_Traffic_Web_Attack.csv')

    # Filter out numerical features for the VAE
    # Dropping non-numeric features like timestamps and IPs for the neural network
    numeric_features = ['bytes_in', 'bytes_out', 'response.code', 'dst_port']
    
    X_normal = normal_df[numeric_features].values
    X_attack = attack_df[numeric_features].values

    print(f"Normal traffic shape: {X_normal.shape}")
    print(f"Attack traffic shape: {X_attack.shape}")

    # Scale the data based strictly on the NORMAL traffic behavior
    scaler = StandardScaler()
    X_normal_scaled = scaler.fit_transform(X_normal)
    X_attack_scaled = scaler.transform(X_attack)

    # Save the scaler for inference in the FastAPI layer later
    os.makedirs('data/processed', exist_ok=True)
    joblib.dump(scaler, 'data/processed/scaler.pkl')

    # Create train and validation splits from normal data
    X_train, X_val = train_test_split(X_normal_scaled, test_size=0.2, random_state=42)
    
    return X_train, X_val, X_attack_scaled, len(numeric_features)


# --- 3. MLOps Experiment Tracking & Training ---

def train_and_track():
    # Setup MLflow
    mlflow.set_experiment("VAE_Web_Attack_Detection")

    # Enable system metrics logging for better observability
    mlflow.enable_system_metrics_logging()
    
    X_train, X_val, X_attack_scaled, input_dim = load_and_preprocess_data()
    
    # Define Hyperparameters
    latent_dim = 2
    epochs = 50
    batch_size = 32
    learning_rate = 0.001

    with mlflow.start_run():
        print("Building Variational Autoencoder...")
        
        # Log params
        mlflow.log_param("latent_dim", latent_dim)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("learning_rate", learning_rate)

        # 1. Build Encoder
        encoder_inputs = keras.Input(shape=(input_dim,))
        x = layers.Dense(16, activation="relu")(encoder_inputs)
        x = layers.Dense(8, activation="relu")(x)
        z_mean = layers.Dense(latent_dim, name="z_mean")(x)
        z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
        z = Sampling()([z_mean, z_log_var])
        encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

        # 2. Build Decoder
        latent_inputs = keras.Input(shape=(latent_dim,))
        x = layers.Dense(8, activation="relu")(latent_inputs)
        x = layers.Dense(16, activation="relu")(x)
        decoder_outputs = layers.Dense(input_dim, activation="linear")(x)
        decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")

        # 3. Instantiate and Compile VAE
        vae = VAE(encoder, decoder)
        vae.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate))

        # Train the model
        print("Training VAE...")
        history = vae.fit(
            X_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, None),
            verbose=1
        )

        # Log training metrics to MLflow
        print("Logging loss curves to MLflow...")
        for epoch in range(epochs):
            mlflow.log_metric("total_loss", history.history["loss"][epoch], step=epoch)
            mlflow.log_metric("reconstruction_loss", history.history["reconstruction_loss"][epoch], step=epoch)
            mlflow.log_metric("kl_loss", history.history["kl_loss"][epoch], step=epoch)
        

        # Calculate dynamic anomaly threshold based on validation normal traffic
        print("Calculating anomaly threshold...")
        reconstructions = vae.predict(X_val)
        
        # Calculate Mean Squared Error per row
        mse_val = np.mean(np.square(X_val - reconstructions), axis=1)
        
        # Threshold dynamic calculation: Mean + 3 Std Devs
        threshold = np.mean(mse_val) + 3 * np.std(mse_val)
        mlflow.log_metric("anomaly_threshold", threshold)
        print(f"Dynamic Threshold established at: {threshold:.4f}")

        # --- Test the model against the real attack data ---
        attack_reconstructions = vae.predict(X_attack_scaled)
        mse_attack = np.mean(np.square(X_attack_scaled - attack_reconstructions), axis=1)
        
        attacks_detected = np.sum(mse_attack > threshold)
        detection_rate = attacks_detected / len(X_attack_scaled) * 100
        
        mlflow.log_metric("attacks_detected_count", attacks_detected)
        mlflow.log_metric("detection_rate_pct", detection_rate)
        
        print(f"\n--- EVALUATION RESULTS ---")
        print(f"Attacks evaluated: {len(X_attack_scaled)}")
        print(f"Attacks flagged as anomalous: {attacks_detected}")
        print(f"Detection Rate: {detection_rate:.2f}%\n")

        # Save the model architecture and weights natively
        os.makedirs('data/processed/vae_model', exist_ok=True)
        vae.save_weights('data/processed/vae_model/vae_weights.weights.h5')
        
        # Log the model artifacts to MLflow
        mlflow.log_artifact('data/processed/scaler.pkl', artifact_path="preprocessor")
        
if __name__ == "__main__":
    train_and_track()