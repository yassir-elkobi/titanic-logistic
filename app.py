import os
import numpy as np
import pandas as pd
from flask import Flask, render_template, request

from model import LogisticRegression
from utils import (
    compute_metrics,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_cost_history,
    plot_kfold_results
)

app = Flask(__name__)


def load_and_preprocess(path):
    """
    Read CSV at `path`, coerce target to 'Survived', select exactly the
    eight fields (fixing 'sibsp'), then fill, one‐hot, scale, and bias-add.
    Returns:
      X, y, feature_names
    """
    import pandas as pd
    import numpy as np

    df = pd.read_csv(path)

    # 1) Rename target and sibsp
    if '2urvived' in df.columns:
        df = df.rename(columns={'2urvived': 'Survived'})
    if 'sibsp' in df.columns:
        df = df.rename(columns={'sibsp': 'SibSp'})

    # 2) Keep only the eight columns
    required = ['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    df = df[required]

    # 3) Fill missing
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    df['Embarked'] = df['Embarked'].fillna('S')

    # 4) One-hot Embarked, binary Sex
    emb = pd.get_dummies(df['Embarked'], prefix='Emb', drop_first=True)
    df = pd.concat([df.drop('Embarked', axis=1), emb], axis=1)
    df['Sex'] = df['Sex'].astype(int)  # already 0/1

    # 5) Split X / y
    y = df['Survived'].values.reshape(-1, 1)
    X = df.drop('Survived', axis=1).values.astype(float)

    # 6) Standardize all features
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma[sigma == 0] = 1.0
    X = (X - mu) / sigma

    # 7) Add bias column
    bias = np.ones((X.shape[0], 1))
    X = np.hstack([bias, X])

    # Final feature names (bias + the rest)
    feature_names = ['Intercept'] + list(df.drop('Survived', axis=1).columns)
    return X, y, feature_names


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Hyperparameters from form
        lr = float(request.form.get("learning_rate", 0.01))
        max_iter = int(request.form.get("max_iterations", 1000))
        lambda_ = float(request.form.get("lambda_", 0.0))
        tolerance = float(request.form.get("tolerance", 1e-4))
        patience = int(request.form.get("patience", 5))
        use_kfold = request.form.get("use_kfold") == "on"
        n_splits = int(request.form.get("n_splits", 5))

        # Load & preprocess
        data_path = os.path.join("static", "train.csv")
        X, y, feature_names = load_and_preprocess(data_path)

        if use_kfold:
            # Manual K-Fold split
            idx = np.random.RandomState(42).permutation(len(y))
            fold_size = len(y) // n_splits
            fold_results = []

            for fold in range(n_splits):
                start = fold * fold_size
                if fold == n_splits - 1:
                    val_idx = idx[start:]
                else:
                    val_idx = idx[start:start + fold_size]
                tr_idx = np.setdiff1d(idx, val_idx)

                X_tr, y_tr = X[tr_idx], y[tr_idx]
                X_val, y_val = X[val_idx], y[val_idx]

                model = LogisticRegression(
                    learning_rate=lr,
                    max_iterations=max_iter,
                    lambda_=lambda_,
                    tolerance=tolerance,
                    patience=patience,
                    verbose=False
                )
                model.fit(X_tr, y_tr, X_val, y_val)

                # Metrics
                train_p = model.predict(X_tr)
                train_pr = model.predict_proba(X_tr)
                val_p = model.predict(X_val)
                val_pr = model.predict_proba(X_val)

                tm = compute_metrics(y_tr, train_p, train_pr)
                vm = compute_metrics(y_val, val_p, val_pr)

                fold_results.append({
                    "fold": fold + 1,
                    "model": model,
                    "train_metrics": tm,
                    "val_metrics": vm,
                    "train_costs": model.cost_history,
                    "val_costs": model.val_cost_history,
                    "y_val": y_val,
                    "val_preds": val_p,
                    "val_probs": val_pr
                })

            # Pick best fold
            best = max(fold_results, key=lambda x: x["val_metrics"]["accuracy"])

            # Plots
            cost_plot = plot_cost_history(best["train_costs"], best["val_costs"])
            cm_plot = plot_confusion_matrix(best["y_val"], best["val_preds"])
            roc_plot, _ = plot_roc_curve(best["y_val"], best["val_probs"])
            kf_plot = plot_kfold_results([
                {"fold": r["fold"], "accuracy": r["val_metrics"]["accuracy"]}
                for r in fold_results
            ])

            results = {
                "mode": "K-Fold CV",
                "n_splits": n_splits,
                "best_fold": best["fold"],
                "train_metrics": best["train_metrics"],
                "val_metrics": best["val_metrics"],
                "cost_plot": cost_plot,
                "cm_plot": cm_plot,
                "roc_plot": roc_plot,
                "kfold_plot": kf_plot
            }

        else:
            # Simple 80/20 split
            rng = np.random.RandomState(42)
            perm = rng.permutation(len(y))
            cut = int(0.8 * len(y))
            tr_idx, val_idx = perm[:cut], perm[cut:]

            X_tr, y_tr = X[tr_idx], y[tr_idx]
            X_val, y_val = X[val_idx], y[val_idx]

            model = LogisticRegression(
                learning_rate=lr,
                max_iterations=max_iter,
                lambda_=lambda_,
                tolerance=tolerance,
                patience=patience,
                verbose=False
            )
            model.fit(X_tr, y_tr, X_val, y_val)

            train_p, train_pr = model.predict(X_tr), model.predict_proba(X_tr)
            val_p, val_pr = model.predict(X_val), model.predict_proba(X_val)

            tm = compute_metrics(y_tr, train_p, train_pr)
            vm = compute_metrics(y_val, val_p, val_pr)

            cost_plot = plot_cost_history(model.cost_history, model.val_cost_history)
            cm_plot = plot_confusion_matrix(y_val, val_p)
            roc_plot, _ = plot_roc_curve(y_val, val_pr)

            results = {
                "mode": "Train/Val Split",
                "train_metrics": tm,
                "val_metrics": vm,
                "cost_plot": cost_plot,
                "cm_plot": cm_plot,
                "roc_plot": roc_plot
            }

        return render_template("results.html", results=results)

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
