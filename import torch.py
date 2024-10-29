import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Create a tensor with several inputs
inputs = torch.linspace(-10, 10, 100)

# Define activation functions
tanh = nn.Tanh()
relu = nn.ReLU()
sigmoid = nn.Sigmoid()
softmax = nn.Softmax(dim=0)

# Apply activation functions
tanh_outputs = tanh(inputs)
relu_outputs = relu(inputs)
sigmoid_outputs = sigmoid(inputs)
softmax_outputs = softmax(inputs)

# Plot the results
plt.figure(figsize=(12, 8))

# TanH
plt.subplot(2, 2, 1)
plt.plot(inputs.numpy(), tanh_outputs.numpy(), label='TanH', color='blue')
plt.title('TanH Activation Function')
plt.xlabel('Input')
plt.ylabel('Output')
plt.legend()

# ReLU
plt.subplot(2, 2, 2)
plt.plot(inputs.numpy(), relu_outputs.numpy(), label='ReLU', color='green')
plt.title('ReLU Activation Function')
plt.xlabel('Input')
plt.ylabel('Output')
plt.legend()

# Sigmoid
plt.subplot(2, 2, 3)
plt.plot(inputs.numpy(), sigmoid_outputs.numpy(), label='Sigmoid', color='red')
plt.title('Sigmoid Activation Function')
plt.xlabel('Input')
plt.ylabel('Output')
plt.legend()

# Softmax
plt.subplot(2, 2, 4)
plt.plot(inputs.numpy(), softmax_outputs.numpy(), label='Softmax', color='purple')
plt.title('Softmax Activation Function')
plt.xlabel('Input')
plt.ylabel('Output')
plt.legend()

plt.tight_layout()
plt.show()
