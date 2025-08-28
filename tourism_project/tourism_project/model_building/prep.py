import pandas as pd
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi
import os

# Authenticate with HF Hub
api = HfApi(token=os.getenv("HF_TOKEN"))

# Load dataset
DATASET_PATH = "hf://datasets/Fitjv/tourism-dataset/tourism_dataset.csv"
tourism_df = pd.read_csv(DATASET_PATH)
print("✅ Dataset loaded successfully.")

target = "ProdTaken"

numeric_features = [
    "Age",
    "DurationOfPitch",
    "NumberOfPersonVisiting",
    "NumberOfFollowups",
    "PreferredPropertyStar",
    "NumberOfTrips",
    "PitchSatisfactionScore",
    "NumberOfChildrenVisiting",
    "MonthlyIncome"
]
categorical_features = [
    "TypeofContact",
    "Occupation",
    "Gender",
    "ProductPitched",
    "MaritalStatus",
    "Designation",
    "Passport",
    "OwnCar",
    "CityTier"
]

X = tourism_df[numeric_features + categorical_features]
y = tourism_df[target]

# Split
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)

# Save locally
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

# Upload splits
for file in ["Xtrain.csv","Xtest.csv","ytrain.csv","ytest.csv"]:
    api.upload_file(
        path_or_fileobj=file,
        path_in_repo=file,
        repo_id="Fitjv/tourism-dataset",
        repo_type="dataset",
    )
print("✅ Train/test splits uploaded to HF")
