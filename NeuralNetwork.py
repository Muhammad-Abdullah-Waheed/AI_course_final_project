import numpy as np
import os
import cv2

# Dataset path
preprocessed_path = "preprocessed_dataset"
class_names = ["sneakers", "sandals", "formal_shoes", "simple_shoes", "slippers"]  # For now, use two classes for binary classification

# Activation function (ReLU and Softmax)
def relu(Z):
    return np.maximum(0, Z)

def softmax(Z):
    expZ = np.exp(Z - np.max(Z, axis=1, keepdims=True))
    return expZ / np.sum(expZ, axis=1, keepdims=True)

# Helper functions for the ANN
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))

def initialize_weights(input_size, hidden_size, output_size):
    np.random.seed(42)  # For reproducibility
    weights = {
        "W1": np.random.randn(input_size, hidden_size) * 0.01,
        "b1": np.zeros((1, hidden_size)),
        "W2": np.random.randn(input_size, hidden_size) * 0.01,
        "b2": np.zeros((1, hidden_size)),
        "W3": np.random.randn(hidden_size, output_size) * 0.01,
        "b3": np.zeros((1, output_size))
    }
    return weights

def forward_propagation(X, weights):
    Z1 = np.dot(X, weights["W1"]) + weights["b1"]
    A1 = relu(Z1)
    Z2 = np.dot(X, weights["W2"]) + weights["b2"]
    A2 = relu(Z1)
    Z3 = np.dot(A1, weights["W3"]) + weights["b3"]
    A3 = softmax(Z2)
    cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2, "Z3": Z3, "A3": A3}
    return A3, cache

def compute_loss(y, A2):
    m = y.shape[0]
    loss = -np.mean(y * np.log(A2) + (1 - y) * np.log(1 - A2))
    return loss

def back_propagation(X, y, weights, cache):
    m = X.shape[0]
    A1 = cache["A1"]
    A2 = cache["A2"]
    A3 = cache["A3"]
    
    dZ2 = A2 - y
    dW2 = np.dot(A1.T, dZ2) / m
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m

    dZ2 = A2 - y
    dW2 = np.dot(A1.T, dZ2) / m
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m

    dZ1 = np.dot(dZ2, weights["W2"].T) * sigmoid_derivative(cache["Z1"])
    dW1 = np.dot(X.T, dZ1) / m
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m

    gradients = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}
    return gradients

def update_weights(weights, gradients, learning_rate):
    weights["W1"] -= learning_rate * gradients["dW1"]
    weights["b1"] -= learning_rate * gradients["db1"]
    weights["W2"] -= learning_rate * gradients["dW2"]
    weights["b2"] -= learning_rate * gradients["db2"]
    return weights

# Load preprocessed images and labels
def load_data():
    data = []
    labels = []

    for label, class_name in enumerate(class_names):
        class_folder = os.path.join(preprocessed_path, class_name)
        
        for image_name in os.listdir(class_folder):
            image_path = os.path.join(class_folder, image_name)
            img = cv2.imread(image_path, cv2.IMREAD_COLOR)
            img = img.flatten()  # Flatten the image
            data.append(img)
            labels.append(label)

    data = np.array(data) / 255.0  # Normalize pixel values
    labels = np.array(labels).reshape(-1, 1)  # Reshape labels to be a column vector
    return data, labels

# Train the ANN
def train(X, y, hidden_size, epochs, learning_rate):
    input_size = X.shape[0]
    output_size = 5  # Five Classes classification
    weights = initialize_weights(input_size, hidden_size, output_size)

    for epoch in range(epochs):
        # Forward propagation
        A2, cache = forward_propagation(X, weights)

        # Compute loss
        loss = compute_loss(y, A2)

        # Backward propagation
        gradients = back_propagation(X, y, weights, cache)

        # Update weights
        weights = update_weights(weights, gradients, learning_rate)

        # Print loss every 100 epochs
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss}")

    return weights

# Predict function
def predict(X, weights):
    A2, _ = forward_propagation(X, weights)
    predictions = (A2 > 0.5).astype(int)
    return predictions

# Main program
if __name__ == "__main__":
    # Load data
    data, labels = load_data()

    # Shuffle data by generating random indices
    indices = np.random.permutation(data.shape[0])

    # Apply shuffled indices to both data and labels
    data_shuffled = data[indices]
    labels_shuffled = labels[indices]

    # Split the data into 80% training and 20% testing
    split_index = int(0.8 * data.shape[0])  # 80% for training
    X_train, y_train = data_shuffled[:split_index], labels_shuffled[:split_index]
    X_test, y_test = data_shuffled[split_index:], labels_shuffled[split_index:]

    # Train the ANN
    hidden_size = 128  # Number of neurons in the hidden layer
    epochs =1000
    learning_rate = 0.01
    weights = train(X_train, y_train, hidden_size, epochs, learning_rate)

    # Evaluate the ANN
    train_predictions = predict(X_train, weights)
    test_predictions = predict(X_test, weights)

    train_accuracy = np.mean(train_predictions == y_train) * 100
    test_accuracy = np.mean(test_predictions == y_test) * 100

    print(f"Training Accuracy: {train_accuracy:.2f}%")
    print(f"Testing Accuracy: {test_accuracy:.2f}%")