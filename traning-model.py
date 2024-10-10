import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, precision_recall_curve, auc
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from imblearn.under_sampling import RandomUnderSampler
from sklearn.neighbors import KNeighborsClassifier

# Load processed data
print("Loading data...")
data_path = '../raw_detection_data/processed_data-drop.csv'
df = pd.read_csv(data_path)
print("Data loaded")

# Extract positive and negative samples
positive_samples = df[df['DOH_Server'] == 1]
negative_samples = df[df['DOH_Server'] == 0]
print(len(positive_samples), len(negative_samples))
# Sample negative samples so that the number is 3 times the positive samples
rus = RandomUnderSampler(sampling_strategy={0: len(positive_samples) * 3, 1: len(positive_samples)}, random_state=42)
X_resampled, y_resampled = rus.fit_resample(df.drop(['IP', 'DOH_Server'], axis=1), df['DOH_Server'])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
print("Train-test split completed")

# Define base models
models = [
    ('rf', RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')),
    ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)),
    # ('catboost', CatBoostClassifier(verbose=0, random_state=42)),
    ('knn', KNeighborsClassifier(n_neighbors=5)), # Decent performance
    # ('lgbm', LGBMClassifier(random_state=42))
]

# Define voting model
voting_model = VotingClassifier(estimators=models, voting='soft')

# Train the model
print("Training voting model...")
voting_model.fit(X_train, y_train)

# Evaluate the model
y_pred_proba = voting_model.predict_proba(X_test)[:, 1]  # Get predicted probabilities for the positive class
y_pred = (y_pred_proba >= 0.4).astype(int)  # Default threshold is 0.5

# Custom decision logic
# Calculate prediction probabilities for each base model
y_pred_probas = np.array([estimator.predict_proba(X_test)[:, 1] for name, estimator in voting_model.named_estimators_.items()])
# If any model predicts a probability greater than 0.4, classify as positive
y_custom_pred = np.any(y_pred_probas > 0.4, axis=0).astype(int)

roc_auc = roc_auc_score(y_test, y_custom_pred)
print(f"Voting Model AUC: {roc_auc:.2f}")
print("Voting Model Classification Report:\n", classification_report(y_test, y_custom_pred))

# Plot ROC curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.figure()
plt.plot(fpr, tpr, label=f'Voting Model (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Voting Model')
plt.legend(loc="lower right")
plt.show()

# Plot PR curve
precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
pr_auc = auc(recall, precision)
plt.figure()
plt.plot(recall, precision, label=f'Voting Model (area = {pr_auc:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve - Voting Model')
plt.legend(loc="lower left")
plt.show()

# Plot confusion matrix
cm = confusion_matrix(y_test, y_custom_pred)
plt.figure(figsize=(6,6))
sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Voting Model')
plt.show()

# Save the model
joblib.dump(voting_model, "./model_storage/Voting_Model.pkl")
print("Voting model saved")

print("All models trained and evaluated")
