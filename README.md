# Crop-Recommendation-Using-ML
A machine learning-based crop recommendation system that predicts the best crops to grow based on environmental factors like temperature, humidity, soil nutrients, and rainfall. The system uses a trained Random Forest model to provide tailored recommendations, helping farmers make data-driven decisions for optimal crop selection.

Here's a sample `README.md` file for your crop recommendation system project:

```markdown
# Crop Recommendation System

## Overview
The **Crop Recommendation System** is a machine learning-based web application designed to help farmers select the best crops to grow based on environmental factors such as temperature, humidity, soil nutrients (Nitrogen, Phosphorus, Potassium), pH levels, and rainfall. The system uses a Random Forest classifier model that has been trained on historical crop data and makes predictions based on user input.

## Features
- **Web Interface**: Users can input environmental parameters and get crop recommendations.
- **Machine Learning Model**: A Random Forest classifier is used to predict the best crop for the given conditions.
- **Scalable Preprocessing**: The system applies Min-Max and Standard Scaling to normalize the input data for better model performance.
- **Model & Scaler Serialization**: The trained model and scalers are saved using `pickle`, allowing the system to load them easily in the Flask application.

## Technologies Used
- **Python**: The primary programming language for data processing and model training.
- **Flask**: A micro web framework used to build the web application.
- **scikit-learn**: A machine learning library used for model training and scaling.
- **pandas**: Used for data manipulation and preprocessing.
- **pickle**: Used for saving and loading the trained model and scalers.
- **HTML/CSS**: Frontend for rendering the user interface.

## Setup Instructions

### Prerequisites
Ensure you have Python installed on your machine. You can download it from [here](https://www.python.org/downloads/).

You also need to install the following dependencies:
```bash
pip install flask pandas numpy scikit-learn
```

### 1. Clone the repository
Clone the repository to your local machine:
```bash
git clone https://github.com/yourusername/Crop-Recommendation-System.git
cd Crop-Recommendation-System
```

### 2. Prepare the environment
Before running the application, make sure you have all necessary files:
- `Crop_recommendation.csv` (Dataset used for training the model)
- `model.pkl` (Trained Random Forest model)
- `minmaxscaler.pkl` (Min-Max Scaler for input normalization)
- `standscaler.pkl` (Standard Scaler for input normalization)

### 3. Train the Model (if not done already)
If you haven't trained the model yet, run the following Python script to train the model and save the necessary files:

```python
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Load the dataset
crop = pd.read_csv("Crop_recommendation.csv")

# Map the crop names to numeric labels
crop_dict = { 'rice': 1, 'maize': 2, 'jute': 3, 'cotton': 4, 'coconut': 5, 'papaya': 6, 'orange': 7, 'apple': 8, 
              'muskmelon': 9, 'watermelon': 10, 'grapes': 11, 'mango': 12, 'banana': 13, 'pomegranate': 14, 
              'lentil': 15, 'blackgram': 16, 'mungbean': 17, 'mothbeans': 18, 'pigeonpeas': 19, 'kidneybeans': 20, 
              'chickpea': 21, 'coffee': 22 }

crop['label'] = crop['label'].map(crop_dict)

X = crop.drop('label', axis=1)
y = crop['label']

# Preprocessing: MinMax and Standard Scaling
mx = MinMaxScaler()
X = mx.fit_transform(X)

sc = StandardScaler()
X = sc.fit_transform(X)

# Train Random Forest Model
randclf = RandomForestClassifier()
randclf.fit(X, y)

# Save the model and scalers using pickle
pickle.dump(randclf, open('model.pkl', 'wb'))
pickle.dump(mx, open('minmaxscaler.pkl', 'wb'))
pickle.dump(sc, open('standscaler.pkl', 'wb'))
```

### 4. Run the Flask Application
After setting up the model and scalers, run the Flask application using the following command:
```bash
python app.py
```

The Flask app will start running on `http://127.0.0.1:5000/` by default.

### 5. Access the Application
Open your web browser and navigate to `http://127.0.0.1:5000/` to use the crop recommendation system. Input the environmental factors and get crop suggestions.

## Usage
1. **Home Page**: The home page allows users to input values for:
   - Nitrogen (N)
   - Phosphorus (P)
   - Potassium (K)
   - Temperature
   - Humidity
   - pH
   - Rainfall
2. After submitting the form, the system will predict the best crop based on the provided values.

## Example
For instance, if you input the following values:
- Nitrogen: 90
- Phosphorus: 42
- Potassium: 43
- Temperature: 20.87
- Humidity: 82.00
- pH: 6.50
- Rainfall: 202.93

The system may predict **Rice** as the most suitable crop for the given conditions.

## Model Accuracy
You can evaluate the accuracy of the model by running it with different classifiers (e.g., Logistic Regression, Naive Bayes, KNN) and comparing the results.

## Contributing
Feel free to fork this repository, make improvements, and submit pull requests. Contributions are welcome!

