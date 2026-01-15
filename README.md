# 🍽️ Food Nutrition Estimator

> **AI-Powered Calorie Prediction from Food Ingredients**  
> A machine learning project that predicts nutritional content using ensemble learning techniques.


## 🎯 Overview

**Food Nutrition Estimator** is an intelligent web application that predicts calorie content from basic food ingredients using Machine Learning. Built with Python and deployed using Streamlit, this project demonstrates practical applications of regression algorithms in nutritional science.

### Why This Project?
- 🏋️ **Health & Fitness**: Help users track their calorie intake
- 🧠 **ML Education**: Showcase regression techniques in real-world scenarios
- 📊 **Data Science**: End-to-end ML pipeline from data generation to deployment

---

## ✨ Features

### 🔮 Smart Prediction
- **Real-time calorie estimation** from 30+ ingredients
- **Interactive ingredient selection** with categorization
- **Quick recipe presets** (Protein Bowl, Salad, Pasta)

### 📊 Data Visualization
- **Pie charts** showing calorie distribution
- **Bar graphs** for ingredient comparison
- **Interactive filters** and search functionality

### 💾 Export & Analysis
- Download nutrition reports as CSV
- Detailed ingredient breakdown
- Calories per 100g calculation

### 🎨 Modern UI
- Gradient design with purple/blue theme
- Responsive layout for all devices
- Intuitive user experience

---

## 💻 Usage

### Quick Start
1. **Select Ingredients**: Choose from 30 common ingredients grouped by category
2. **Enter Amounts**: Input ingredient quantities in grams
3. **View Results**: Get instant calorie predictions with visual breakdowns
4. **Download Report**: Export your nutrition data as CSV

### Example Recipe
```python
Chicken Breast: 150g
Rice: 100g
Broccoli: 80g
Olive Oil: 10g
-------------------
Total Calories: ~450 kcal
```

### API Usage (Python)
```python
import joblib
import numpy as np

# Load model
model = joblib.load('nutrition_model.pkl')
scaler = joblib.load('scaler.pkl')

# Prepare ingredients (in grams)
ingredients = {
    'chicken_breast': 150,
    'rice': 100,
    'broccoli': 80,
    'olive_oil': 10
}

# Predict
features = [ingredients.get(ing, 0) for ing in ingredient_list]
features_scaled = scaler.transform(np.array(features).reshape(1, -1))
calories = model.predict(features_scaled)[0]

print(f"Predicted Calories: {calories:.0f} kcal")
```

---

## 📈 Model Performance

### Dataset Statistics
- **Total Samples**: 500 food recipes
- **Features**: 30 ingredients (amounts in grams)
- **Target**: Total calories (kcal)

### Models Compared
| Model | MAE (calories) | RMSE (calories) | R² Score |
|-------|----------------|-----------------|----------|
| Linear Regression | 12.45 | 18.32 | 0.9823 |
| Ridge Regression | 12.38 | 18.21 | 0.9825 |
| **Lasso Regression** | **11.92** | **17.84** | **0.9831** |
| Random Forest | 13.15 | 19.05 | 0.9812 |
| Gradient Boosting | 12.87 | 18.76 | 0.9818 |

### Best Model: **Lasso Regression**
- ✅ **98.3% accuracy** (R² Score)
- ✅ Average error of only **~12 calories**
- ✅ Robust to overfitting with L1 regularization

---

## 🛠️ Tech Stack

### Machine Learning
- **Scikit-learn**: Model training and evaluation
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations

### Web Application
- **Streamlit**: Interactive web interface
- **Plotly**: Data visualization and charts

### Development Tools
- **Jupyter Notebook**: Exploratory data analysis
- **Joblib**: Model serialization
- **Python 3.8+**: Core programming language

---

## 📁 Project Structure

```
food-nutrition-estimator/
│
├── app.py                      # Streamlit web application
├── create_dataset.py           # Dataset generation script
├── train_model.py             # Quick model training script
├── nutrition_model.ipynb      # Detailed training notebook
│
├── nutrition_model.pkl        # Trained ML model
├── scaler.pkl                 # Feature scaler
├── ingredients.json           # Ingredient list
├── food_nutrition_data.csv    # Generated dataset
│
├── README.md                  # Project documentation
└── LICENSE                    # MIT License
```

---
