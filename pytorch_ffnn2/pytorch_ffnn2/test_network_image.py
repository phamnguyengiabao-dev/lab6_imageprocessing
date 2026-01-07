import torch
import torch.nn as nn
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from layer_simple import FCLayer, ActivationLayer
from function_simple import Activation, ActivationPrime, Loss, LossPrime
from network_simple import Network
import numpy as np

# 1. Load Data
print("Loading MNIST data...")
transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: torch.flatten(x))])

# Download and load training data
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Use DataLoader for batching
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

# 2. Visualize Data (Reused logic)
def visualize_data(loader):
    data_iter = iter(loader)
    images, labels = next(data_iter)
    
    figure = plt.figure(figsize=(8, 8))
    cols, rows = 3, 3
    for i in range(1, cols * rows + 1):
        if i > len(images): break
        img = images[i-1].reshape(28, 28)
        label = labels[i-1].item()
        figure.add_subplot(rows, cols, i)
        plt.title(f"Label: {label}")
        plt.axis("off")
        plt.imshow(img, cmap="gray")
    plt.show()

print("Visualizing input data...")
visualize_data(train_loader)

# 3. Model Setup
# The generic Network class in Sample #2 allows adding layers dynamically.
# FCLayer(in_size, out_size)
# ActivationLayer(activation, derivative)

net = Network()
net.add(FCLayer(28 * 28, 128))               # Input to Hidden
net.add(ActivationLayer(Activation.tanh, ActivationPrime.tanh_derivative))
net.add(FCLayer(128, 10))                    # Hidden to Output
net.add(ActivationLayer(Activation.tanh, ActivationPrime.tanh_derivative))

# 4. Visualize Architecture
def visualize_model(model):
    print("\nModel Architecture:")
    for i, layer in enumerate(model.layers):
        print(f"Layer {i+1}: {type(layer).__name__}")
        if hasattr(layer, 'weights'):
            print(f"  Weights: {layer.weights.shape}")

visualize_model(net)

# 5. Training Setup
net.use(Loss.mse, LossPrime.mse_prime)

print("\nStarting Training...")
epochs = 5
learning_rate = 0.1

# The `net.fit` method in Sample #2 takes full x_train and y_train tensors loop internally?
# Let's check `network.py`: yes, `fit(self, x_train, y_train, epochs, alpha)`
# and it iterates `epochs`, then `samples`. 
# THIS IS VERY SLOW for full MNIST (60k samples).
# ALSO it expects input to be list of tensors or tensor [Samples, Features] ?
# From test_network_simple.py: x_train = torch.tensor(...) shape [4, 1, 2] ? No.
# x_train = torch.tensor([[[0, 0]], [[0, 1]]...]) -> Shape [4, 1, 2]
# y_train = torch.tensor([[[0]], [[1]]...]) -> Shape [4, 1, 1]
# So it expects [Batch/Sample, 1, Feature] format for single sample processing inside loop?
# Inner loop: `for k in range(samples): output = x_train[k] ...`
# So x_train[k] should be the input vector for one sample.
# We should reshape our data to match this or modify `fit` to handle batches.
# Given strict instruction to "Modify sample", better to adapt data to sample's expectation if possible, 
# or copy-paste logic and adapt for batch if too slow. 
# For demonstration, we can use a smaller subset of MNIST to keep it fast enough for this naive implementation.

subset_size = 1000
print(f"Using subset of {subset_size} samples for training to fit naive generic implementation speed.")

# Prepare data in correct shape [N, 1, 784] and [N, 1, 10]
data_iter = iter(torch.utils.data.DataLoader(train_dataset, batch_size=subset_size, shuffle=True))
images, labels = next(data_iter)

# Reshape input: [N, 784] -> [N, 1, 784]
x_train = images.unsqueeze(1) 

# One-hot encode targets
y_onehot = torch.zeros(labels.size(0), 10)
y_onehot.scatter_(1, labels.view(-1, 1), 1)
# Reshape target: [N, 10] -> [N, 1, 10]
y_train = y_onehot.unsqueeze(1)

# Train
net.fit(x_train, y_train, epochs=epochs, alpha=learning_rate)

# 6. Prediction
print("\nTesting prediction...")
test_images, test_labels = next(iter(test_loader))
x_test = test_images[0].unsqueeze(0).unsqueeze(0) # [1, 1, 784]
test_label = test_labels[0].item()

output = net.predicts(x_test)[0] # Returns list of outputs, get first
predicted_label = torch.argmax(output).item()

print(f"True Label: {test_label}")
print(f"Predicted Label: {predicted_label}")
print(f"Output: {output}")
