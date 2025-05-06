import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.3, random_state=42
)

# Set experiment name (you can see this in MLflow UI)
mlflow.set_experiment("iris_random_forest")

with mlflow.start_run():
    # Train model
    clf = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42)
    clf.fit(X_train, y_train)

    # Predict and evaluate
    predictions = clf.predict(X_test)
    acc = accuracy_score(y_test, predictions)

    # Log parameters and metrics
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 3)
    mlflow.log_metric("accuracy", acc)

    # Log the model itself
    mlflow.sklearn.log_model(clf, "random_forest_model")

    print(f"Model accuracy: {acc}")
