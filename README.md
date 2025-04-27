# Titanic Survival Predictor with Logistic Regression

**Titanic Survival Predictor** is a Flask web application that predicts the survival probability of Titanic passengers
based on their characteristics. It implements logistic regression _from scratch_ using gradient descent (instead of
relying on scikit-learn for model training). This project is part of my machine learning journey, aiming to solidify
understanding of classification algorithms, gradient descent optimization, and end-to-end model deployment.

## Features

* **Custom Logistic Regression Implementation**
* **Batch Gradient Descent Optimization**
* **L2 Regularization**
* **Early Stopping on Validation Loss**
* **K-Fold Cross-Validation**
* **Comprehensive Evaluation Metrics**
* **Interactive Visualizations**
* **Numerical Stability Improvements**

## Tech Stack and Tools

* **Python 3**
* **Flask**
* **NumPy**
* **Pandas**
* **Matplotlib**
* **tqdm (for progress visualization)**
* **scikit-learn (for metrics calculation only)**

## Best Model Configuration

Through experimentation, the following configuration was found to produce the best results for the Titanic survival
prediction task:

* **Algorithm:** Logistic Regression with Batch Gradient Descent
* **Regularization:** L2 (Ridge) to prevent overfitting
* **Cross-Validation:** K-Fold (k=5) for robust model evaluation
* **Early Stopping:** Enabled with patience=10 for optimal training duration
* **Gradient Descent Parameters:** Learning Rate α = 0.05, Number of Iterations = 2000

## Dataset Summary

The model is trained on the famous Titanic dataset, which contains information about passengers aboard the RMS Titanic,
including whether they survived the ship's sinking. The dataset includes about 891 passengers with the following key
features:

* **Pclass:** Passenger class (1st, 2nd, or 3rd)
* **Sex:** Gender of the passenger
* **Age:** Age of the passenger
* **SibSp:** Number of siblings/spouses aboard
* **Parch:** Number of parents/children aboard
* **Fare:** Passenger fare
* **Embarked:** Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)

The **target** variable is **Survived** (0 = No, 1 = Yes), indicating whether the passenger survived the disaster.

## Project Structure

```
├── app.py                 # Main Flask application
├── model.py               # Logistic regression model implementation
├── preprocessing.py       # Data loading and preprocessing functions
├── utils.py               # Evaluation metrics and visualization utilities
├── requirements.txt       # Project dependencies
├── static/                # Static files (CSS, data files)
│   └── train.csv          # Titanic dataset
└── templates/             # HTML templates
    ├── index.html         # Settings page
    └── results.html       # Results visualization page
```

## How to Run the App Locally

1. **Clone the repository** or download the project source code to your local machine.
2. **Create a virtual environment:**
   ```
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
3. **Install Python dependencies:**
   ```
   pip install -r requirements.txt
   ```
4. **Run the Flask application:**
   ```
   python app.py
   ```
5. **Open the web app in your browser:** Navigate to http://127.0.0.1:5000/

## Usage

1. Adjust model parameters on the settings page:
    - Learning rate and number of iterations
    - L2 regularization strength
    - Early stopping settings
    - K-fold cross-validation options

2. Click "Train & Evaluate" to build and analyze the model

3. View comprehensive results:
    - Training and validation metrics
    - Learning curves
    - Confusion matrix
    - ROC curve
    - Cross-validation performance (if enabled)

## Learning Goals

This project was developed as part of my journey to learn and demonstrate machine learning engineering skills. The
primary learning goal was to **implement logistic regression from scratch** and integrate it into a full web
application.

**Note:** This README is written to be clear and informative for any new visitor (or recruiter) checking out the
project. It highlights the project’s purpose, capabilities, technical implementation, and the learning outcomes
associated with it.