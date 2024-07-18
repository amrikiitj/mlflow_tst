import os
import sklearn
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import logging
import mlflow
import mlflow.sklearn

# Enable logging for debugging
logging.basicConfig(level=logging.DEBUG)

# Set MLflow tracking URI to a specific subdirectory in the current directory
mlflow_tracking_dir = os.path.join(os.getcwd(), "mlruns")
os.makedirs(mlflow_tracking_dir, exist_ok=True)
mlflow.set_tracking_uri("file://" + mlflow_tracking_dir)
mlflow.set_experiment("my_experiment")

tr_size = 0.7

data = make_regression(n_samples=1000, shuffle=True, random_state=22)
x = data[0]
y = data[1]
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=tr_size)
model = LinearRegression()

with mlflow.start_run() as run:
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    mean_sq = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    # Log parameters and metrics
    mlflow.log_param("train_size", tr_size)
    mlflow.log_metric("MSE", mean_sq)
    mlflow.log_metric("R2", r2)

    # Infer and log model signature
    signature = mlflow.models.signature.infer_signature(x_test, predictions)
    logging.debug(f"Model signature: {signature}")
    mlflow.sklearn.log_model(model, "lr_model", signature=signature)

    # Log artifact
    artifact_path = os.path.join(os.getcwd(), "artifacts")
    os.makedirs(artifact_path, exist_ok=True)
    artifact_file = os.path.join(artifact_path, "example.txt")
    # with open(artifact_file, "w") as f:
    #     f.write("This is an example artifact.")
    # mlflow.log_artifact(artifact_file)

    # Log the run ID for debugging
    run_id = run.info.run_id
    logging.debug(f"Run ID: {run_id}")

print("Run completed.")
