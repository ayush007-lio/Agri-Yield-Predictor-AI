# Agri-Yield-Predictor-AI
# 🌾 Smart Crop Recommendation System

### 🚜 Precision Agriculture using Machine Learning (KNN)

**Domain:** Agriculture / Socio-Economic Development  
**Tech Stack:** Python, Scikit-Learn, Pandas, Numpy

---

## 📖 Project Overview
Agriculture is the backbone of the economy, yet many farmers suffer losses due to selecting crops that are mismatched with their soil profile or climate. 

This project solves this problem by building a **Smart Recommendation System**. It uses the **K-Nearest Neighbors (KNN)** algorithm to analyze 7 key biological parameters (Nitrogen, Phosphorus, Potassium, pH, Temperature, Humidity, Rainfall) and predicts the most suitable crop for a specific farm to maximize yield.

## 🎯 Societal & Government Impact
* **Doubling Farmers' Income:** By suggesting the highest-yield crop, we reduce the risk of crop failure.
* **Soil Health Preservation:** Prevents nutrient depletion caused by planting the wrong crops repeatedly.
* **Data-Driven Farming:** Shifts agriculture from "guesswork" to "precision science."

## 🧠 Technical Implementation
The core logic relies on finding statistical similarities between a new farm's conditions and historical successful harvests.

### The Algorithm: K-Nearest Neighbors (KNN)
I chose KNN because agriculture relies on natural clusters. If a specific N-P-K ratio works for "Rice" in one dataset entry, it will likely work for a test case with similar values.

### Feature Scaling (Critical Step)
The dataset contains features with vastly different ranges:
* **Rainfall:** 0 - 300 mm
* **Soil pH:** 0 - 14
* **Potassium:** 0 - 200

If raw data is fed into KNN, "Rainfall" would dominate the distance calculation simply because the numbers are bigger. I implemented **`StandardScaler`** to normalize all features to a mean of 0 and variance of 1, ensuring the model treats Soil pH as equally important as Rainfall.

## 📂 Project Structure
```text
Crop_Recommender/
│
├── data/
│   └── Crop_recommendation.csv   # (Download from Kaggle)
│
├── src/
│   └── app.py                    # Main application (Training + Prediction)
│
├── requirements.txt              # Dependencies
└── README.md                     # Documentation
