import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import joblib
import os
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("mlops-tourism-experiment")

api = HfApi()

# Paths on Hugging Face datasets repo
Xtrain_path = "hf://datasets/Fitjv/tourism-dataset/tourism_train.csv"
Xtest_path  = "hf://datasets/Fitjv/tourism-dataset/tourism_test.csv"

# Load train/test
train_df = pd.read_csv(Xtrain_path)
test_df  = pd.read_csv(Xtest_path)

# Separate features and target
target = "ProdTaken"
Xtrain = train_df.drop(columns=[target])
ytrain = train_df[target]

Xtest = test_df.drop(columns=[target])
ytest = test_df[target]

# Define features
numeric_features = ["Age", "MonthlyIncome", "DurationOfPitch", "NumberOfTrips"]
categorical_features = ["Gender", "Occupation", "MaritalStatus"]

# Handle class imbalance
class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]

# Preprocessing
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown='ignore'), categorical_features)
)

# Base model
xgb_model = xgb.XGBClassifier(scale_pos_weight=class_weight, random_state=42)

# Hyperparameter grid
param_grid = {
    'xgbclassifier__n_estimators': [50, 75, 100],
    'xgbclassifier__max_depth': [2, 3, 4],
    'xgbclassifier__learning_rate': [0.01, 0.1],
    'xgbclassifier__reg_lambda': [0.4, 0.6]
}

# Pipeline
model_pipeline = make_pipeline(preprocessor, xgb_model)

# Train with MLflow logging
with mlflow.start_run():
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, n_jobs=-1)
    grid_search.fit(Xtrain, ytrain)

    # Log results
    results = grid_search.cv_results_
    for i in range(len(results['params'])):
        param_set = results['params'][i]
        mean_score = results['mean_test_score'][i]
        with mlflow.start_run(nested=True):
            mlflow.log_params(param_set)
            mlflow.log_metric("mean_test_score", mean_score)

    # Log best params
    mlflow.log_params(grid_search.best_params_)
    best_model = grid_search.best_estimator_

    # Evaluate
    y_pred_test = best_model.predict(Xtest)
    test_report = classification_report(ytest, y_pred_test, output_dict=True)

    mlflow.log_metrics({
        "test_accuracy": test_report['accuracy'],
        "test_precision": test_report['1']['precision'],
        "test_recall": test_report['1']['recall'],
        "test_f1": test_report['1']['f1-score']
    })

    # Save model
    model_path = "tourism_model_xgb.joblib"
    joblib.dump(best_model, model_path)
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"✅ Model saved: {model_path}")

    # Upload to Hugging Face model hub
    repo_id = "Fitjv/tourism-model"
    try:
        api.repo_info(repo_id=repo_id, repo_type="model")
        print(f"Repo '{repo_id}' already exists.")
    except RepositoryNotFoundError:
        print(f"Repo '{repo_id}' not found. Creating...")
        create_repo(repo_id=repo_id, repo_type="model", private=False)

    api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo=model_path,
        repo_id=repo_id,
        repo_type="model"
    )
    print(f"✅ Model uploaded to Hugging Face: {repo_id}")
