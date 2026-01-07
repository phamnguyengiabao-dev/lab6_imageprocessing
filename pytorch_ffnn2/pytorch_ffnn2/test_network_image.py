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
net.add(FCLayer(28 * 28, 128))               # Input [Batch, 784] -> Hidden [Batch, 128]
net.add(ActivationLayer(Activation.tanh, ActivationPrime.tanh_derivative))
net.add(FCLayer(128, 10))                    # Hidden [Batch, 128] -> Output [Batch, 10]
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

print("\n" + "="*30)
print("TRAINING PROGRESS")
print("="*30)
epochs = 5
learning_rate = 0.1

subset_size = None # Use Full Dataset
if subset_size:
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
else:
    print("Using FULL MNIST dataset with Mini-Batch Training.")
    # For full dataset, we should load it all into memory if it fits (60k is fine)
    # Or modify fit to take loader. 
    # Current fit takes tensors. Let's load full dataset into tensors.
    
    loader_full = torch.utils.data.DataLoader(train_dataset, batch_size=60000, shuffle=True)
    images, labels = next(iter(loader_full))
    
    # [60000, 784] -> [60000, 1, 784] 
    # Actually, `fc_layer` does matmul. 
    # If Input is [Batch, 1, 784] -> FC -> [Batch, 1, Hidden].
    # But conventionally we use [Batch, 784].
    # Let's check `fc_layer.py`: `self.out_data = torch.matmul(self.in_data, self.weights) + self.bias`
    # Weights [In, Out].
    # If InData [Batch, In] -> Out [Batch, Out].
    # This is correct. We do NOT need the extra dimension 1 if we process strictly in batches.
    # BUT, `forward` in `network.py` does loop layers.
    # Wait, the previous failing implementation used `x_train[k]` which was a single vector.
    # If we pass `x_batch` of shape `[32, 784]`, then `fc_layer` logic holds.
    # So we should pass `x_train` as `[N, 784]`, NOT `[N, 1, 784]`.
    # Let's correct this.
    
    x_train = images # [60000, 784]
    
    y_onehot = torch.zeros(labels.size(0), 10)
    y_onehot.scatter_(1, labels.view(-1, 1), 1)
    y_train = y_onehot # [60000, 10]

# Train
# Now using batch_size=64
net.fit(x_train, y_train, epochs=epochs, alpha=learning_rate, batch_size=64)

# 6. Final Evaluation
print("\n" + "="*30)
print("FINAL EVALUATION")
print("="*30)
print("Testing prediction on separate test set (1 item)...")
# Note: Predicting 10k items with this naive implementation in loop might be slow.
# Let's assess accuracy on a subset of Test data (e.g. 100 items)
test_subset_size = 100
print(f"Evaluating on random {test_subset_size} test samples...")

test_loader_subset = torch.utils.data.DataLoader(test_dataset, batch_size=test_subset_size, shuffle=True)
test_images, test_labels = next(iter(test_loader_subset))

# Reshape for prediction loop: [N, 784] list or tensor
# We updated training to use [N, 784], so prediction should also use [N, 784]
x_test_batch = test_images # [N, 784]

outputs = net.predicts(x_test_batch)

correct = 0
for i in range(len(outputs)):
    # Output of layer is [10] or [1, 10] depending on if predict loop processes one by one
    # predict in network.py:
    # for i in range(samples): output = data[i]; ...
    # data[i] from [N, 784] is [784].
    # Forward pass on [784]: MatMul([784], [784, 128]) -> error if 1D?
    # PyTorch handles 1D matmul automatically? 
    # torch.matmul(vector, matrix) -> vector.
    # So output will be [128].
    # Finally [10].
    
    pred = torch.argmax(outputs[i])
    true = test_labels[i]
    if pred == true:
        correct += 1

print(f"Test Subset Accuracy: {100 * correct / test_subset_size:.2f}% ({correct}/{test_subset_size})")

print("-" * 20)
print("Single Prediction Demo:")
x_single = test_images[0] # [784]
output_single = net.predict(x_single) # Use single predict method
predicted_label = torch.argmax(output_single).item()
true_label = test_labels[0].item()

print(f"True Label:      {true_label}")
print(f"Predicted Label: {predicted_label}")
print(f"Confidence:      {torch.max(output_single).item():.4f}")
print("="*30)
