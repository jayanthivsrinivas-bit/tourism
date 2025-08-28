import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from huggingface_hub import hf_hub_download

# Load test data
test_df = pd.read_csv("hf://datasets/Fitjv/tourism-dataset/tourism_test.csv")
Xtest = test_df.drop(columns=["ProdTaken"])
ytest = test_df["ProdTaken"]

# Load model from Hugging Face
model_path = hf_hub_download(repo_id="Fitjv/tourism-model", filename="tourism_model_xgb.joblib")
model = joblib.load(model_path)

# Predictions
y_pred = model.predict(Xtest)

# Metrics
report = classification_report(ytest, y_pred, output_dict=True)
print("✅ Evaluation Report:", report)

# Save report as CSV
report_df = pd.DataFrame(report).transpose()
report_df.to_csv("evaluation_results.csv", index=True)

# Confusion matrix
cm = confusion_matrix(ytest, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Not Taken","Taken"], yticklabels=["Not Taken","Taken"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("confusion_matrix.png")
print("✅ Confusion matrix saved")
