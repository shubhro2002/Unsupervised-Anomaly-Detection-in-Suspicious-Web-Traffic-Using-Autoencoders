# Web Traffic Anomaly Detection and MLOps Threat Intelligence Pipeline

## Overview

This project represents a complete architectural evolution of an unsupervised anomaly detection system for cybersecurity. The initial version relied on a basic Autoencoder trained solely on malicious network traffic due to the unavailability of normal traffic logs. 

Version 2.0 transforms this baseline into a production-ready MLOps pipeline. It solves the cold-start data problem using Generative AI, upgrades the core model to a Variational Autoencoder (VAE), implements rigorous experiment tracking, serves the model via a REST API, and integrates a privacy-first Retrieval-Augmented Generation (RAG) pipeline for automated incident response.

The goal of this system is to ingest high-dimensional web traffic metrics, detect zero-day attacks or WAF-bypassing volumetric anomalies in real-time, and automatically generate actionable, plain-English security alerts for SOC (Security Operations Center) analysts without exposing sensitive infrastructure data to the public internet.

## Key Improvements from Version 1.0

* Data Strategy Shift: Moved from training on attack data to training on synthetic "normal" baseline data, treating the actual CloudWatch attack logs as the holdout test set for true anomaly detection.
* Algorithm Upgrade: Replaced the standard Autoencoder with a Variational Autoencoder (VAE) to enforce a continuous, structured latent space, resulting in highly accurate anomaly thresholding.
* MLOps Integration: Integrated MLflow for comprehensive experiment tracking, model artifact versioning, and system hardware profiling.
* Real-Time Serving: Wrapped the trained model and preprocessing scalers into a FastAPI application for live HTTP inference.
* Automated Threat Intelligence: Built a local LLM-powered RAG pipeline (Ollama/Llama-3) utilizing query transformation to translate mathematical anomalies into semantic search queries and generate SOC reports.

## System Architecture

The pipeline is divided into five distinct operational phases:

### 1. Synthetic Data Generation (Cold-Start Resolution)
Because the original AWS CloudWatch dataset only contained malicious traffic, we utilized the Gemini API to generate a mathematically consistent synthetic dataset of "normal" web traffic. This dataset adheres to natural distributions of bytes inbound, bytes outbound, and response codes, establishing a clean baseline for the model to learn from.

### 2. Variational Autoencoder (VAE) Training
The core detection engine is a custom VAE built with TensorFlow and Keras. 
* It learns to compress and reconstruct normal web traffic features.
* The loss function combines Mean Squared Error (Reconstruction Loss) and KL Divergence (Latent Space Regularization).
* The anomaly threshold is dynamically calculated based on the validation set's mean reconstruction error plus three standard deviations.

### 3. MLOps Experiment Tracking
The training module is fully integrated with MLflow.
* Automatically logs hyperparameters (latent dimension, batch size, learning rate).
* Tracks epoch-by-epoch convergence of total loss, reconstruction loss, and KL loss.
* Versions model artifacts, including the specific StandardScaler used during training, preventing data leakage in production.

### 4. Real-Time Inference Layer
A FastAPI REST application serves the trained VAE. It exposes a /predict endpoint that accepts raw web log JSON payloads, scales the features, reconstructs them through the VAE, compares the MSE against the dynamic threshold, and returns a binary anomaly classification instantly.

### 5. Threat Intelligence and RAG Pipeline
When an anomaly is detected, the raw numeric metrics undergo Query Transformation. High-dimensional data (e.g., bytes_in > 30000) is converted into semantic text (e.g., "massive inbound payload"). This query can be used to retrieve historical CVEs or attack patterns from a vector database. A local Llama-3 model (via Ollama) then processes this context to generate a concise incident response alert.

## Results and Evaluation

The VAE was trained entirely on the synthetic normal baseline and tested against the original 282 raw CloudWatch attack logs. 
* Detection Rate: The model successfully flagged ~97.5% of the real-world attacks as anomalies based purely on their deviation from the learned normal latent space.
* Thresholding: The dynamic thresholding mechanism successfully separated standard web traffic fluctuations from genuine volumetric attacks.

## Production and Data Privacy Considerations

To ensure the architecture aligns with enterprise security and data residency compliance, the following protocols are designed into the system:

* Cryptographic Anonymization: All raw IP addresses are hashed using a salted SHA-256 algorithm before being passed to downstream analytics or the LLM. This allows analysts to track repeat offenders without logging raw PII.
* Infrastructure Isolation: By utilizing Ollama to run Llama-3 locally, the Threat Intelligence pipeline is entirely air-gapped. Network topology and security logs never traverse the public internet, ensuring compliance with strict data privacy frameworks.
* Data Minimization Strategy: If deployed, raw logs processed by the FastAPI layer should be held in an ephemeral cache (e.g., Redis) with a strict Time-To-Live (TTL) deletion policy, as the mathematical weights of the VAE do not require permanent raw data storage.

## Repository Setup and Execution

To run this pipeline locally:

### 1. Environment Setup
Install dependencies from the requirements file.
```bash
pip install -r requirements.txt

### 2. MLOps Tracking Server
Start the MLflow UI in a dedicated terminal instance.
```bash
mlflow ui --port 5000

## 3. Model Training
Execute the training script to build the VAE and calculate the threshold.
```bash
python src/train_autoencoder.py

## 4. API Serving
Start the FastAPI server for real-time inference.
```bash
uvicorn src.serve_model:app --reload

Access the Swagger UI at http://127.0.0.1:8000/docs

## 5. Threat Intelligence Generation
Ensure Ollama is installed and the Llama-3 model is running (`ollama run llama3`). Execute the threat intelligence script to simulate an anomaly detection event and generate the SOC report.
```bash
python src/threat_intel_rag.py