import torch
import torch.nn as nn
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from nn_simple import ffnn
import numpy as np

# 1. Load Data
print("Loading MNIST data...")
transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: torch.flatten(x))])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Use a subset for faster demonstration if needed, but full dataset is fine for simple NN
# Taking a small batch for manual training loop demonstration similar to original sample
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

# 2. Visualize Data
def visualize_data(loader):
    data_iter = iter(loader)
    images, labels = next(data_iter)
    
    figure = plt.figure(figsize=(8, 8))
    cols, rows = 3, 3
    for i in range(1, cols * rows + 1):
        if i > len(images): break
        img = images[i-1].reshape(28, 28) # Unflatten for visualization
        label = labels[i-1].item()
        figure.add_subplot(rows, cols, i)
        plt.title(f"Label: {label}")
        plt.axis("off")
        plt.imshow(img, cmap="gray")
    plt.show()

print("Visualizing input data...")
visualize_data(train_loader)

# 3. Model Setup
input_size = 28 * 28 # 784
hidden_size = 128
output_size = 10 

NN = ffnn.FFNeuralNetwork(inputSize=input_size, hiddenSize=hidden_size, outputSize=output_size)

# 4. Visualize Architecture
def visualize_model(model):
    print("\nModel Architecture:")
    print(f"Input Layer: {model.inputSize} neurons")
    print(f"Hidden Layer: {model.hiddenSize} neurons")
    print(f"Output Layer: {model.outputSize} neurons")
    print("\nLayer Weights Shapes:")
    print(f"W1: {model.W1.shape}")
    print(f"W2: {model.W2.shape}")

visualize_model(NN)

# 5. Training Loop
# The original generic FFNN implementation expects manual backprop.
# We need to adapt the training loop to use One-Hot encoding for MSE Loss as used in original sample
# or modify the backward pass. The original backward pass is hardcoded for specific loss derivative?
# Original: out_error = y - o. This implies MSE with linear output or Sigmoid+MSE.
# W2 update: self.W2 += torch.matmul(torch.t(self.z2), self.out_delta) * rate where out_delta = error * sigmoid_derivative
# This is standard backprop for MSE loss with Sigmoid activation.

# 5. Training Loop
print("\n" + "="*30)
print("TRAINING PROGRESS")
print("="*30)
epochs = 5
learning_rate = 0.1

for epoch in range(epochs):
    total_loss = 0
    correct = 0
    total = 0
    
    for i, (images, labels) in enumerate(train_loader):
        # Flatten input (done in transform)
        X = images
        
        # One-hot encode targets
        y_onehot = torch.zeros(labels.size(0), output_size)
        y_onehot.scatter_(1, labels.view(-1, 1), 1)
        
        # Forward
        output = NN.forward(X)
        
        # Calculate Loss (MSE)
        loss = torch.mean((y_onehot - output) ** 2)
        total_loss += loss.item()
        
        # Calculate Accuracy
        predicted = torch.argmax(output, dim=1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        
        # Backward and Optimize (Manual)
        NN.backward(X, y_onehot, output, learning_rate)
        
        if (i+1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] | Step [{i+1}/{len(train_loader)}] | Loss: {loss.item():.4f}")
            
    epoch_acc = 100 * correct / total
    print(f">> Epoch {epoch+1} Summary: Avg Loss: {total_loss / len(train_loader):.4f} | Accuracy: {epoch_acc:.2f}%")

# 6. Save Weights
NN.save_weights(NN, "NN_MNIST")
print("\n[INFO] Weights saved to 'NN_MNIST'")

# 7. Final Evaluation
print("\n" + "="*30)
print("FINAL EVALUATION")
print("="*30)
print("Testing prediction on separate test set...")

# Evaluate on full test set
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        # Batch size is 1 from loader definition, but let's handle generic
        X = images
        output = NN.forward(X)
        predicted = torch.argmax(output, dim=1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

print(f"Test Set Accuracy: {100 * correct / total:.2f}% ({correct}/{total})")

# Single sample demo
print("-" * 20)
print("Single Prediction Demo:")
images, labels = next(iter(test_loader))
x_predict = images[0]
label = labels[0].item()

output = NN.forward(x_predict.unsqueeze(0))
predicted_label = torch.argmax(output).item()

print(f"True Label:      {label}")
print(f"Predicted Label: {predicted_label}")
print(f"Confidence:      {torch.max(output).item():.4f}")
print("="*30)
