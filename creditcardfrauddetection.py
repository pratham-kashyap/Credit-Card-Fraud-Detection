import numpy as np
import pandas as pd

data = pd.read_csv('credit_card_data.csv')

X = data.drop('Class', axis=1)
y = data['Class']             

X = (X - X.mean()) / X.std()
X['bias'] = 1

X = X.to_numpy()
y = y.to_numpy()

num_training_samples = 199367
X_train, X_test = X[:num_training_samples], X[num_training_samples:]
y_train, y_test = y[:num_training_samples], y[num_training_samples:]

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

num_features = X_train.shape[1]
theta = np.zeros(num_features)
learning_rate = 0.01
num_epochs = 900

for epoch in range(num_epochs):
    predictions = sigmoid(np.dot(X_train, theta))
    
    error = -y_train * np.log(predictions) - (1 - y_train) * np.log(1 - predictions)
    cost = np.mean(error)
    
    gradient = np.dot(X_train.T, (predictions - y_train)) / len(y_train)
    
    theta -= learning_rate * gradient
    
    if (epoch + 1) % 100 == 0:
        print(f'Epoch {epoch + 1}/{num_epochs}, Cost: {cost}')
        predictions_test = sigmoid(np.dot(X_test, theta))
        threshold = 0.5
        predicted_labels_test = (predictions_test >= threshold).astype(int)
        # Calculate the evaluation metrics from scratch
        accuracy_test = np.mean(predicted_labels_test == y_test)
        precision_test = np.sum((y_test == 1) & (predicted_labels_test == 1)) / np.sum(predicted_labels_test == 1)
        recall_test = np.sum((y_test == 1) & (predicted_labels_test == 1)) / np.sum(y_test == 1)
        f1_score_test = 2 * (precision_test * recall_test) / (precision_test + recall_test)
        # Print the evaluation metrics
        print("Accuracy:", accuracy_test)
        print("Precision:", precision_test)
        print("Recall:", recall_test)
        print("F1 Score:", f1_score_test)
        print()