from sklearn import svm
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import IsolationForest


data = pd.read_csv('/Users/canhalukferhatoglu/Downloads/training_data_with_ids_and_anomalies.csv')

data['timestamp'] = pd.to_datetime(data['timestamp'])
data['hour'] = data['timestamp'].dt.hour
data['day_of_week'] = data['timestamp'].dt.dayofweek
data['day_of_month'] = data['timestamp'].dt.day

# features and target variable
features = data[['latitude', 'longitude', 'direction', 'hour', 'day_of_week', 'day_of_month']]
labels = data['is_anomaly']

# Scale the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Split data into training and test sets
train_features = scaled_features[labels == 0]
train_labels = labels[labels == 0]

test_features = scaled_features
test_labels = labels

# Train SVM model
print("One-Class SVM Model Outcome:")
print(" ")
model = svm.OneClassSVM(kernel='rbf', gamma='scale', nu=0.05)
model.fit(train_features)

# Predict anomalies
predictions = model.predict(test_features)
# Map predictions to 0 (normal) and 1 (anomaly)
predictions = np.where(predictions == -1, 1, 0)

# Evaluate the model
print("Confusion Matrix:")
print(confusion_matrix(test_labels, predictions))
print("\nClassification Report:")
print(classification_report(test_labels, predictions))


# Split data into normal and anomaly subsets
normal_data = data[labels == 0]
anomaly_data = data[labels == 1]

# Randomly sample a subset of anomalies
sampled_anomalies = anomaly_data.sample(frac=0.5, random_state=42)

# Combine normal data with sampled anomalies
train_data_subset = pd.concat([normal_data, sampled_anomalies])

# features and target variable
train_features_subset = train_data_subset[['latitude', 'longitude', 'direction', 'hour', 'day_of_week', 'day_of_month']]
train_labels_subset = train_data_subset['is_anomaly']

scaled_train_features_subset = scaler.fit_transform(train_features_subset)

# Train Isolation Forest using subset
print("Isolation Forest Model Outcome with Subset:")
print(" ")
iso_forest = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
iso_forest.fit(scaled_train_features_subset)

# Predict anomalies for the entire dataset
iso_predictions = iso_forest.predict(test_features)
iso_predictions = np.where(iso_predictions == -1, 1, 0)

# Evaluate Isolation Forest 
print("Confusion Matrix (Isolation Forest with Subset):")
print(confusion_matrix(test_labels, iso_predictions))
print("\nClassification Report (Isolation Forest with Subset):")
print(classification_report(test_labels, iso_predictions))
