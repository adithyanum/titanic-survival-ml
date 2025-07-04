# 🛳️ Titanic Survival Prediction Using Machine Learning

This repository contains a complete end-to-end machine learning pipeline to predict whether a passenger survived the Titanic disaster, based on features such as age, class, fare, and gender.

---

## 📌 Project Objective

The goal is to apply supervised machine learning techniques to predict the survival of Titanic passengers using classical data preprocessing, model training, hyperparameter tuning, and evaluation.

This project is part of my learning journey as an aspiring AI engineer, showcasing a full machine learning workflow with explanations at each step.

---

## 🧠 What You’ll Learn

- How to clean and preprocess structured data
- Encoding categorical variables
- Train-test split and cross-validation
- Training a Random Forest model
- Hyperparameter tuning using GridSearchCV
- Evaluating model performance using confusion matrix and classification report
- Saving and loading trained models with `joblib`

---

## 📂 Project Structure

| Section                          | Description |
|----------------------------------|-------------|
| `Titanic_Prediction_ML.ipynb`    | Main Jupyter/Colab notebook with all code and markdown explanations |
| `titanic_random_forest_model.pkl`| Saved model file using `joblib` (optional) |
| `README.md`                      | Project summary and structure |

---

## 📊 Tools and Libraries Used

- Python 🐍
- Pandas
- NumPy
- Seaborn & Matplotlib
- Scikit-learn (RandomForestClassifier, GridSearchCV, Metrics)
- Joblib

---

## 📈 Final Accuracy

Achieved an accuracy of **~87%** using a tuned Random Forest classifier with selected features and proper encoding. Evaluated using a confusion matrix and F1-score.

---

## 💾 How to Use the Model

To load and use the trained model:

```python
import joblib
model = joblib.load('titanic_random_forest_model.pkl')
predictions = model.predict(X_test)
