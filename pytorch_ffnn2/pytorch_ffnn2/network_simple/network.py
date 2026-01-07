from function_simple import Loss, LossPrime
import torch


class Network:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_prime = None

    def add(self, layer):
        self.layers.append(layer)

    def use(self, loss: Loss, loss_prime: LossPrime) -> None:
        self.loss = loss
        self.loss_prime = loss_prime

    # forward propagation
    def predict(self, data):
        output = data
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def predicts(self, data):
        samples = len(data)
        result = []

        # for every input vector data x_i do:
        for i in range(samples):
            # forward propagation
            output = data[i]
            for layer in self.layers:
                output = layer.forward(output)
            result.append(output)
        return result

    def fit(self, x_train, y_train, epochs, alpha, batch_size=32):
        samples = len(x_train)
        if batch_size is None:
            batch_size = samples

        # for every training cycle
        for i in range(epochs):
            total_error = 0
            
            # Mini-batch training
            for j in range(0, samples, batch_size):
                # Get mini-batch
                # Note: inputs should be [Batch, Features], outputs [Batch, OutputDim]
                x_batch = x_train[j:j+batch_size]
                y_batch = y_train[j:j+batch_size]
                
                # Check dimensions for FCLayer (expects [Batch, In])
                # Current layers handle broadcasting or matmul correctly if inputs are 2D.
                
                # Forward propagation
                output = x_batch
                for layer in self.layers:
                    output = layer.forward(output)
                
                # Calculate error (MSE for batch)
                # self.loss returns element-wise error? Or scalar?
                # fn_simple calls: torch.mean((y_true - y_pred) ** 2) usually.
                # But here we need the derivative `loss_prime`.
                
                # Accumulate for display (scalar)
                # Assuming loss returns mean or sum. Let's trust generic loss for now.
                batch_loss = self.loss(y_batch, output)
                # If batch_loss is a tensor, take item.
                if isinstance(batch_loss, torch.Tensor):
                    total_error += batch_loss.item()
                else:
                    total_error += batch_loss

                # Backward propagation
                gradient = self.loss_prime(y_batch, output)
                for layer in reversed(self.layers):
                    gradient = layer.backward(gradient, alpha)

            # Average error per batch or per sample? 
            # If loss is Mean, then sum of means / num_batches is average epoch loss.
            num_batches = (samples + batch_size - 1) // batch_size
            avg_error = total_error / num_batches
            print(f'Epoch {i+1}/{epochs} error = {avg_error:.6f}')
