import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Load and preprocess the data
data = pd.read_csv('creditcard.csv')
print("Columns in the DataFrame:", data.columns)
print("First few rows of the DataFrame:")
print(data.head())

# Use 'Class' as the label column
X = data.drop(columns=['Class']).values  # Drop the 'Class' column to get features
y = data['Class'].values  
# Encode labels (assuming 'normal' is 0 and 'anomaly' is 1)
y = (y == 'anomaly').astype(int)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the data
X_train_mean = X_train.mean(axis=0)
X_train_std = X_train.std(axis=0)
X_train = (X_train - X_train_mean) / X_train_std
X_test = (X_test - X_train_mean) / X_train_std

# Define the autoencoder architecture
input_dim = X_train.shape[1]
encoding_dim = 32

# Input layer
input_layer = Input(shape=(input_dim,))
# Encoder
encoded = Dense(encoding_dim, activation='relu')(input_layer)
# Decoder
decoded = Dense(input_dim, activation='sigmoid')(encoded)

# Autoencoder model
autoencoder = Model(input_layer, decoded)
autoencoder.compile(optimizer=Adam(), loss='mse')

# Train the autoencoder
autoencoder.fit(X_train, X_train, epochs=100, batch_size=256, shuffle=True, validation_data=(X_test, X_test))

# Calculate reconstruction error
reconstructed = autoencoder.predict(X_test)
reconstruction_error = np.mean(np.abs(reconstructed - X_test), axis=1)

# Set a threshold (this can be fine-tuned)
threshold = np.percentile(reconstruction_error, 90)
print(f"Threshold for anomaly detection: {threshold}")

# Identify anomalies
anomalies = reconstruction_error > threshold

# Print the number of detected anomalies
print(f"Number of detected anomalies: {np.sum(anomalies)}")
print(f"Number of true anomalies in y_test: {np.sum(y_test)}")

# Evaluation metrics
precision = np.sum(anomalies[y_test == 1]) / np.sum(anomalies)
recall = np.sum(anomalies[y_test == 1]) / np.sum(y_test)
f1 = 2 * (precision * recall) / (precision + recall)

print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')

# Evaluation metrics
num_anomalies_detected = np.sum(anomalies)
num_true_anomalies = np.sum(y_test)

# Calculate precision
if num_anomalies_detected > 0:
    precision = np.sum(anomalies[y_test == 1]) / num_anomalies_detected
else:
    precision = 0.0  # No anomalies detected

# Calculate recall
if num_true_anomalies > 0:
    recall = np.sum(anomalies[y_test == 1]) / num_true_anomalies
else:
    recall = 0.0  # No true anomalies in the test set

# Calculate F1 Score
if (precision + recall) > 0:
    f1 = 2 * (precision * recall) / (precision + recall)
else:
    f1 = 0.0  # No precision or recall

print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')


# Visualization
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.hist(reconstruction_error[y_test == 0], bins=50, alpha=0.5, label='Normal Transactions')
plt.hist(reconstruction_error[y_test == 1], bins=50, alpha=0.5, label='Fraudulent Transactions')
plt.axvline(x=threshold, color='r', linestyle='--', label='Threshold')
plt.xlabel('Reconstruction Error')
plt.ylabel('Frequency')
plt.title('Reconstruction Error Distribution')
plt.legend()
plt.show()
