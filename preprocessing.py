import pandas as pd
import numpy as np


def load_data(path):
    """
    Load raw Titanic CSV into a DataFrame.
    """
    df = pd.read_csv(path)
    return df


def preprocess_data(df):
    """
    Manual feature engineering & scaling without sklearn:
      - Titles from Name
      - Deck from Cabin
      - FamilySize, IsAlone
      - Age and Fare bins
      - One-hot encoding for categorical
      - Manual standardization for continuous
    Returns:
      X, y, feature_names, (mu, sigma) for scaled cols if needed
    """
    df = df.copy()

    # 1) Titles
    df['Title'] = df['Name'].str.split(',').str[1].str.split('.').str[0].str.strip()
    rare = df['Title'].value_counts()[lambda x: x < 10].index
    df['Title'] = df['Title'].replace(rare, 'Rare')
    title_dummies = pd.get_dummies(df['Title'], prefix='Title')

    # 2) Deck
    df['Deck'] = df['Cabin'].fillna('U').str[0]
    deck_dummies = pd.get_dummies(df['Deck'], prefix='Deck')

    # 3) Family
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

    # 4) Age & Fare fill
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())

    # 5) Age binning
    df['AgeBin'] = pd.cut(df['Age'], bins=[0, 12, 20, 40, 60, 100], labels=False)
    age_dummies = pd.get_dummies(df['AgeBin'], prefix='AgeBin')

    # 6) Fare binning
    df['FareBin'] = pd.qcut(df['Fare'], 4, labels=False)
    fare_dummies = pd.get_dummies(df['FareBin'], prefix='FareBin')

    # 7) Embarked & Sex
    df['Embarked'] = df['Embarked'].fillna('S')
    emb_dummies = pd.get_dummies(df['Embarked'], prefix='Emb')
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

    # Combine and drop unused cols
    model_df = pd.concat([
        df[['Survived', 'Pclass', 'Sex', 'FamilySize', 'IsAlone']],
        title_dummies, deck_dummies, age_dummies, fare_dummies, emb_dummies
    ], axis=1)

    # Standardize continuous: FamilySize, Age, Fare
    for col in ['FamilySize', 'Age', 'Fare']:
        mu, sd = model_df[col].mean(), model_df[col].std()
        model_df[col] = (model_df[col] - mu) / sd

    # Prepare arrays
    y = model_df['Survived'].values.reshape(-1, 1)
    X = model_df.drop('Survived', axis=1).values
    # add bias term
    X = np.hstack([np.ones((X.shape[0], 1)), X])
    feature_names = ['Intercept'] + model_df.drop('Survived', axis=1).columns.tolist()

    return X, y, feature_names


def create_k_folds(X, y, n_splits=5, random_state=42):
    """
    Manual K-fold split returning list of (X_train, X_val, y_train, y_val).
    """
    np.random.seed(random_state)
    idx = np.random.permutation(len(y))
    fold_size = len(y) // n_splits
    folds = []
    for i in range(n_splits):
        start = i * fold_size
        if i == n_splits - 1:
            val_idx = idx[start:]
        else:
            val_idx = idx[start:start + fold_size]
        train_idx = np.setdiff1d(idx, val_idx)
        folds.append((X[train_idx], X[val_idx], y[train_idx], y[val_idx]))
    return folds
