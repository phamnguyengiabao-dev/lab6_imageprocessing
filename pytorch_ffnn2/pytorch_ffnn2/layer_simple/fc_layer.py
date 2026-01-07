from layer_simple.base_layer import BaseLayer

import torch


class FCLayer(BaseLayer):
    def __init__(self, in_size, out_size):
        self.in_data = None
        self.out_data = None

        # He initialization or Xavier would be better, but for simplicity:
        # Scale down significantly to avoid immediate Tanh saturation
        self.weights = torch.randn(in_size, out_size) * 0.01
        self.bias = torch.randn(1, out_size) * 0.01

    def forward(self, in_data):
        self.in_data = in_data
        self.out_data = torch.matmul(self.in_data, self.weights) + self.bias

        return self.out_data

    def backward(self, out_error, rate):
        in_error = torch.matmul(out_error, self.weights.T)
        weights_error = torch.matmul(self.in_data.T, out_error)

        # For batch training, we sum the gradients
        # weights_error is already [In, Batch] @ [Batch, Out] = [In, Out] (Sum)
        
        # Bias gradient should be sum of out_error along batch dimension
        bias_error = torch.sum(out_error, dim=0, keepdim=True)

        # Update weights (Normalize by batch size? 
        # Original code didn't normalize (batch=1).
        # If we sum, effective learning rate is rate * batch_size.
        # It's safer to not change the "rate" definition but the user optimization request implies standard behavior.
        # But if I change rate logic, I deviate from original simple implementation logic.
        # However, without normalization, training might explode with Batch=64.
        # Let's normalize by input sample count (batch size).
        
        batch_size = out_error.shape[0]
        
        self.weights -= (rate / batch_size) * weights_error
        self.bias -= (rate / batch_size) * bias_error

        return in_error
