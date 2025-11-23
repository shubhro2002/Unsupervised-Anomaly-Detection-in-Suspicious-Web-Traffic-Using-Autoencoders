# Unsupervised-Anomaly-Detection-in-Suspicious-Web-Traffic-Using-Autoencoders

## Overview

This project focuses on building an autoencoder-based anomaly detection model to learn latent patterns in malicious web traffic. The dataset provided contains only attack traffic, with no examples of normal user activity.

The goal is:

- To learn the internal structure of known attack samples.

- To identify deviations from learned patterns as potential new or unknown attacks.

- To build a pipeline for anomaly detection that can later be extended once normal traffic becomes available.

This approach is widely used in cybersecurity for detecting zero-day attacks, fraud, and behavioral anomalies when clean data is limited or unavailable.

## Dataset

File: `CloudWatch_Traffic_Web_Attack.csv`

Rows: ~282

Type: Only malicious (attack) network traffic

The dataset includes:

- Network flow attributes

- Source & destination information

- Volume indicators

- Time-based features

- AWS CloudWatch metadata

- Attack labels & rule information

Since it contains no normal traffic, the project is structured as a purely unsupervised anomaly detection task.

## Objectives

- Perform exploratory data analysis (EDA) on attack traffic.

- Select meaningful behavioral features and remove identifying metadata.

- Preprocess and scale the data.

- Build and train an autoencoder model to learn attack patterns.

- Generate reconstruction errors that can be used for anomaly scoring.

- Prepare the pipeline for future evaluation once normal traffic becomes available.


## Why use Autoencoder?

Autoencoders are ideal for this task because they:

- Learn a compressed representation of attack behavior

- Reconstruct samples similar to training data with low error

- Produce high reconstruction errors for anomalous patterns

## Data Preprocessing Pipeline

1️. Drop Non-Behavioral Metadata

Removed columns that leak attack labels or have no behavioral significance:

- `rule_names`

- `observation_name`

- `source.meta`

- `source.name`

- `detection_types`

- `src_ip`, `dst_ip`

These columns would cause the autoencoder to memorize attack types instead of learning traffic patterns.

2️. Feature Selection

Selected 19 behavioral features such as:

- `bytes_in`, `bytes_out`

- `response.code`

- `dst_port`, `protocol`

- `src_ip_country_code`

Time features:

- `creation_hour`, `creation_minute`, `creation_weekday`

- `end_hour`, `end_minute`, `end_weekday`

- `Epoch timestamps`

- `Connection duration`

These represent true web traffic behavior.

3️. Encoding & Scaling

- Categorical encoding for features like `protocol`

- Standardization using `StandardScaler`

- Train/validation split: `80%` / `20%`

## Autoencoder Model Architecture

A symmetric deep autoencoder:

Input Layer (19 features)

↓

Dense(32, relu)

↓

Dense(16, relu)

↓

Dense(8, relu)     ← Bottleneck

↓

Dense(16, relu)

↓

Dense(32, relu)

↓

Dense(19, linear)


Optimization:

- Loss: MSE (Mean Squared Error)

- Optimizer: Adam

- Epochs: 50–100

- Batch size: 16

The bottleneck learns core attack patterns.

## Training

- Trained only on attack data

- Validation loss monitored

- Training curves used to verify non-overfitting

Once trained, the model is able to:

- Reconstruct attack samples accurately

- Produce reconstruction errors for each sample

## Reconstruction Error Analysis

The distribution of reconstruction errors reveals:

- Typical attack behavior (low error)

- Outliers within attack traffic (high error)

- Basis for anomaly thresholding

## Future Work

When normal traffic becomes available:

1. Evaluate anomaly detection performance using:

   - ROC-AUC

   - Precision-Recall

   - F1 score

   - Confusion Matrix

2. Retrain autoencoder on:

   - Only normal traffic (classic one-class AE), or

   - Combined mixed traffic with semi-supervised learning

3. Implement online anomaly detection for real-time monitoring.

## Conclusion

This project successfully demonstrates:

- Unsupervised anomaly detection on malicious traffic

- Autoencoder-based behavior learning

- Detection of deviations and potential new attacks

- A scalable pipeline that can integrate normal traffic later

Even without normal samples, the results are meaningful, valid, and industry-relevant for anomaly-based intrusion detection.
