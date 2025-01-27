import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split

class SimpleNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        """Initialize neural network with random weights"""
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Initialize weights and biases
        self.W1 = np.random.randn(self.input_size, self.hidden_size) * 0.01
        self.b1 = np.zeros((1, self.hidden_size))
        self.W2 = np.random.randn(self.hidden_size, self.output_size) * 0.01
        self.b2 = np.zeros((1, self.output_size))

    def sigmoid(self, x):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        """Derivative of sigmoid function"""
        return x * (1 - x)

    def forward(self, X):
        """Forward propagation"""
        # First layer
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        
        # Output layer
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        
        return self.a2

    def backward(self, X, y, output):
        """Backward propagation"""
        self.output_error = y - output
        self.output_delta = self.output_error * self.sigmoid_derivative(output)

        self.z2_error = np.dot(self.output_delta, self.W2.T)
        self.z2_delta = self.z2_error * self.sigmoid_derivative(self.a1)

        # Update weights and biases
        self.W2 += np.dot(self.a1.T, self.output_delta)
        self.b2 += np.sum(self.output_delta, axis=0, keepdims=True)
        self.W1 += np.dot(X.T, self.z2_delta)
        self.b1 += np.sum(self.z2_delta, axis=0, keepdims=True)

    def train(self, X, y, epochs, learning_rate=0.01):
        """Train the neural network"""
        self.loss_history = []
        
        for epoch in range(epochs):
            # Forward propagation
            output = self.forward(X)
            
            # Backward propagation
            self.backward(X, y, output)
            
            # Calculate loss
            loss = np.mean(np.square(y - output))
            self.loss_history.append(loss)
            
            # Print progress
            if epoch % 1000 == 0:
                print(f'Epoch {epoch}, Loss: {loss:.4f}')

    def predict(self, X):
        """Make predictions"""
        return (self.forward(X) > 0.5).astype(int)

def create_dataset():
    """Create a simple dataset using make_circles"""
    X, y = make_circles(n_samples=1000, noise=0.1, factor=0.3)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train.reshape(-1, 1), y_test.reshape(-1, 1)

def plot_decision_boundary(model, X, y):
    """Plot the decision boundary"""
    h = 0.01
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Decision Boundary')

def main():
    # Create dataset
    print("Creating dataset...")
    X_train, X_test, y_train, y_test = create_dataset()

    # Initialize neural network
    print("\nInitializing neural network...")
    input_size = 2
    hidden_size = 4
    output_size = 1
    nn = SimpleNeuralNetwork(input_size, hidden_size, output_size)

    # Train the model
    print("\nTraining neural network...")
    nn.train(X_train, y_train, epochs=5000, learning_rate=0.1)

    # Make predictions
    print("\nMaking predictions...")
    train_predictions = nn.predict(X_train)
    test_predictions = nn.predict(X_test)

    # Calculate accuracy
    train_accuracy = np.mean(train_predictions == y_train)
    test_accuracy = np.mean(test_predictions == y_test)
    print(f'\nTraining Accuracy: {train_accuracy:.4f}')
    print(f'Testing Accuracy: {test_accuracy:.4f}')

    # Plot results
    plt.figure(figsize=(15, 5))
    
    # Plot training curve
    plt.subplot(1, 2, 1)
    plt.plot(nn.loss_history)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Time')
    
    # Plot decision boundary
    plt.subplot(1, 2, 2)
    plot_decision_boundary(nn, X_train, y_train)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()