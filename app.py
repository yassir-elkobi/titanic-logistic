import os
import io
import base64
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from flask import Flask, render_template, request

##########################
# 1. FLASK SETUP         #
##########################
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret')  # Move to env in prod


##########################
# 1. DATA LOADING        #
##########################
def load_data(path):
    """Load Titanic CSV and select features + target."""
    df = pd.read_csv(path)
    # Rename the '2urvived' column to 'Survived' if it exists, otherwise use as is
    if '2urvived' in df.columns:
        df = df.rename(columns={'2urvived': 'Survived'})
    
    # Ensure columns exist and have correct types before selection
    columns_to_select = ['Survived', 'Pclass', 'Sex', 'Age', 'sibsp', 'Parch', 'Fare', 'Embarked']
    
    # Create a copy before modifying to avoid chained assignment warnings
    df_selected = df[columns_to_select].copy()
    
    # Rename 'sibsp' to 'SibSp' for consistency
    df_selected = df_selected.rename(columns={'sibsp': 'SibSp'})
    
    # Fill NA values without using inplace
    df_selected['Age'] = df_selected['Age'].fillna(df_selected['Age'].median())
    
    # Convert Embarked to string type before filling with 'S'
    df_selected['Embarked'] = df_selected['Embarked'].astype(str).fillna('S')
    
    # One-hot encode categorical
    df_selected = pd.get_dummies(df_selected, columns=['Sex', 'Embarked'], drop_first=True)
    
    # Ensure all columns are numeric for logistic regression
    for col in df_selected.columns:
        if col != 'Survived':  # Keep target as is
            df_selected[col] = pd.to_numeric(df_selected[col], errors='coerce')
    
    # Fill any potential NaN values created during conversion
    df_selected = df_selected.fillna(0)
    
    return df_selected


##########################
# 2. PREPROCESSING       #
##########################
def standard_scale(X):
    """Z-score normalization."""
    return (X - X.mean()) / X.std()


##########################
# 3. LOGISTIC REGRESSION #
##########################
def sigmoid(z):
    """Compute sigmoid value for array z."""
    # Ensure z is a numpy array
    z = np.array(z, dtype=np.float64)
    return 1 / (1 + np.exp(-z))


def compute_cost(X, y, weights, lambda_):
    """
    Binary cross-entropy cost with L2 regularization.
    J = -1/m * [y log(h) + (1-y) log(1-h)] + (λ/2m) * ||w||^2
    """
    # Ensure all inputs are numpy arrays with proper dtypes
    X = np.array(X, dtype=np.float64)
    y = np.array(y, dtype=np.float64)
    weights = np.array(weights, dtype=np.float64)
    lambda_ = float(lambda_)
    
    m = len(y)
    h = sigmoid(X.dot(weights))
    # Avoid log(0)
    epsilon = 1e-8
    cost = - (1 / m) * (y.T.dot(np.log(h + epsilon)) + (1 - y).T.dot(np.log(1 - h + epsilon)))
    reg = (lambda_ / (2 * m)) * np.sum(np.square(weights[1:]))  # exclude bias
    
    # Convert to scalar if possible
    result = float(cost + reg)
    return result


def gradient_descent(X, y, weights, learning_rate, num_iters, lambda_):
    """
    Perform gradient descent to learn weights.
    dW = (1/m) * Xᵀ(h - y) + (λ/m)*w
    """
    m = len(y)
    # Ensure all inputs are numpy arrays with proper dtypes
    X = np.array(X, dtype=np.float64)
    y = np.array(y, dtype=np.float64)
    weights = np.array(weights, dtype=np.float64)
    learning_rate = float(learning_rate)
    lambda_ = float(lambda_)
    
    cost_history = []
    for i in range(num_iters):
        h = sigmoid(X.dot(weights))
        error = h - y
        grad = (1 / m) * (X.T.dot(error))
        # regularize all but bias term
        grad[1:] += (lambda_ / m) * weights[1:]
        # Ensure grad is float64 before subtraction
        grad = np.array(grad, dtype=np.float64)
        weights = weights - learning_rate * grad  # explicit subtraction instead of -=
        cost = compute_cost(X, y, weights, lambda_)
        cost_history.append(cost)
    
    return weights, cost_history


def predict(X, weights):
    """Return binary predictions 0/1."""
    # Ensure inputs are numpy arrays with proper dtypes
    X = np.array(X, dtype=np.float64)
    weights = np.array(weights, dtype=np.float64)
    
    probs = sigmoid(X.dot(weights))
    return (probs >= 0.5).astype(int)


##########################
# 4. FLASK ROUTES        #
##########################
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Hyperparameters
        lr = float(request.form['learning_rate'])
        iters = int(request.form['num_iters'])
        use_reg = request.form.get('regularization') == 'on'
        lambda_ = float(request.form['lambda']) if use_reg else 0.0

        # Load & preprocess
        df = load_data(os.path.join('static', 'train.csv'))
        y = df.pop('Survived').values.reshape(-1, 1)
        X = df.values
        X = standard_scale(X)  # scale features
        m, n = X.shape
        X = np.hstack([np.ones((m, 1)), X])  # add bias term

        # Initialize weights
        weights = np.zeros((n + 1, 1))

        # Train
        weights, cost_hist = gradient_descent(X, y, weights, lr, iters, lambda_)

        # Predict & evaluate
        preds = predict(X, weights)
        accuracy = np.mean(preds == y) * 100

        # Plot cost history
        buf = io.BytesIO()
        plt.figure()
        plt.plot(cost_hist)
        plt.title('Cost over Iterations')
        plt.xlabel('Iteration')
        plt.ylabel('Cost')
        plt.tight_layout()
        plt.savefig(buf, format='png');
        plt.close()
        cost_plot = base64.b64encode(buf.getvalue()).decode()

        return render_template('results.html',
                               accuracy=accuracy,
                               cost_plot=cost_plot)
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
