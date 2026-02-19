from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

def train_model(df, target_column, algorithm):

    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Convert text columns automatically
    X = X.apply(lambda col: col.astype('category').cat.codes if col.dtype == 'object' else col)
    y = y.astype('category').cat.codes

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    if algorithm == "Decision Tree":
        model = DecisionTreeClassifier()

    elif algorithm == "Naive Bayes":
        model = GaussianNB()

    else:
        model = SVC()

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    return accuracy_score(y_test, preds)
