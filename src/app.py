import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def load_and_train():
    print("🌱 Loading Crop Data...")
    try:
        # Load the dataset
        df = pd.read_csv('data/Crop_recommendation.csv')
    except FileNotFoundError:
        print("Error: 'data/Crop_recommendation.csv' not found.")
        print("Please download it from Kaggle and put it in the data folder.")
        exit()

    # 1. Separate Features (Soil/Weather) and Target (Crop Name)
    # N - Nitrogen, P - Phosphorus, K - Potassium
    X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
    y = df['label']

    # 2. Scale the features
    # CRITICAL: N is 0-140, pH is 0-14. KNN needs them on the same scale (0-1).
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 3. Train/Test Split (80% for training, 20% for testing)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # 4. Train the KNN Model
    # K=5 is usually a good balance for this dataset
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)

    # 5. Check Accuracy
    predictions = knn.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    print(f"✅ Model Trained! Accuracy: {acc*100:.2f}%")
    
    return knn, scaler

def predict_crop(model, scaler):
    print("\n--- 🌾 FARMER'S ASSISTANT TOOL 🌾 ---")
    print("Enter the soil and weather details below:")
    
    try:
        # Get user input
        n = float(input("Nitrogen (N) content (e.g., 90): "))
        p = float(input("Phosphorus (P) content (e.g., 40): "))
        k = float(input("Potassium (K) content (e.g., 40): "))
        temp = float(input("Temperature (°C) (e.g., 20): "))
        hum = float(input("Humidity (%) (e.g., 80): "))
        ph = float(input("Soil pH (0-14) (e.g., 6.5): "))
        rain = float(input("Rainfall (mm) (e.g., 200): "))
        
        # Preprocess the input just like the training data
        user_input = np.array([[n, p, k, temp, hum, ph, rain]])
        user_input_scaled = scaler.transform(user_input)
        
        # Predict
        prediction = model.predict(user_input_scaled)
        
        print(f"\n🌟 Recommended Crop: {prediction[0].upper()} 🌟")
        
        # Optional: Show 'neighbors' logic
        print("(Based on similarity to other successful farms with these conditions)")

    except ValueError:
        print("❌ Invalid input. Please enter numbers only.")

if __name__ == "__main__":
    # Train the model once when the program starts
    model, scaler = load_and_train()
    
    # Keep the tool running
    while True:
        predict_crop(model, scaler)
        cont = input("\nTry another soil profile? (y/n): ")
        if cont.lower() != 'y':
            print("Goodbye! Happy Farming. 🚜")
            break