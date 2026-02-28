import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder

def train_model(df, target_column, algorithm):
    # 1. Drop rows where the Target is missing (we can't train on nothing)
    df = df.dropna(subset=[target_column])
    
    # 2. Separate Features (X) and Target (y)
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # 3. Fill missing values in Features (Prevent crashes)
    # Filling with 0 is a safe default for a generic demo app
    X = X.fillna(0)

    # 4. Encode Categorical Features (Strings -> Numbers)
    # We loop through every column. If it's text, we turn it into numbers.
    for col in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        # Convert to string first to handle mixed types safely
        X[col] = le.fit_transform(X[col].astype(str))

    # 5. Encode Target if it is text
    if y.dtype == 'object':
        le_target = LabelEncoder()
        y = le_target.fit_transform(y.astype(str))

    # 6. Split Data (70% Train, 30% Test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # 7. Select Algorithm
    if algorithm == "Decision Tree":
        model = DecisionTreeClassifier()
    elif algorithm == "Naive Bayes":
        model = GaussianNB()
    elif algorithm == "SVM":
        model = SVC()
    else:
        # Default fallback
        model = DecisionTreeClassifier()

    # 8. Train and Predict
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    return accuracy_score(y_test, preds)