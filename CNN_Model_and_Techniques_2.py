import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import sys
import numpy as np

# Step 10: Apply Grad-CAM Technique

# Explicitly specify download=True and provide a manual path
train_dataset = datasets.MNIST(
    root='./data',
    train=True,
    download=True,  # This should trigger alternative download methods
    transform=transforms.ToTensor()
)


class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1)
        self.fc1 = nn.Linear(32 * 5 * 5, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Safer hook registration with error handling
        try:
            def backward_hook(module, grad_input, grad_output):
                self.gradients = grad_output[0] if grad_output else None

            def forward_hook(module, input, output):
                self.activations = output

            self.backward_handle = target_layer.register_backward_hook(backward_hook)
            self.forward_handle = target_layer.register_forward_hook(forward_hook)
        except Exception as e:
            print(f"Hook registration error: {e}")

    def __del__(self):
        # Ensure hooks are removed to prevent memory leaks
        if hasattr(self, 'backward_handle'):
            self.backward_handle.remove()
        if hasattr(self, 'forward_handle'):
            self.forward_handle.remove()

    def generate_cam(self, input_image, target_class=None):
        # Comprehensive error-proof CAM generation
        try:
            # Ensure input is detached and requires gradients
            input_image = input_image.detach().clone().requires_grad_(True)

            # Temporarily set model to train mode for gradient tracking
            self.model.train()

            # Forward pass with explicit gradient tracking
            with torch.enable_grad():
                model_output = self.model(input_image)

                # Reset gradients
                self.model.zero_grad()
                input_image.grad = None

                # Determine target class
                if target_class is None:
                    target_class = model_output.argmax().item()

                # Create one-hot encoded target
                one_hot = torch.zeros_like(model_output)
                one_hot[0][target_class] = 1

                # Backward pass with comprehensive error handling
                model_output.backward(gradient=one_hot, retain_graph=True)

            # Safely extract and process gradients
            if self.gradients is None:
                print("No gradients captured. Skipping CAM generation.")
                return np.zeros((7, 7), dtype=np.float32)

            # Pooled gradients computation
            pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])

            # Weight activations
            for i in range(self.activations.size(1)):
                self.activations[:, i, :, :] *= pooled_gradients[i]

            # Generate CAM
            cam = torch.mean(self.activations, dim=1).squeeze().detach()
            cam = F.relu(cam)

            # Normalize with small epsilon to prevent division by zero
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

            return cam.cpu().numpy()

        except Exception as e:
            print(f"CAM generation error: {e}")
            return np.zeros((7, 7), dtype=np.float32)

def train_and_visualize():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)

    model = SimpleNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop (same as before)
    epochs = 3
    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

        print(f"Epoch {epoch+1}: Loss {total_loss/len(train_loader):.4f}, Accuracy {100. * correct / total:.2f}%")

    # Visualization with comprehensive error handling
    print("\n\n\n \033[1mGradCAM\033[0m \n")
    model.eval()
    gradcam = GradCAM(model, model.conv2)

    plt.figure(figsize=(15, 10))

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            for i in range(min(5, len(data))):
                img = data[i].unsqueeze(0)
                pred_output = model(img)

                true_label = target[i].item()
                pred_label = pred_output.argmax().item()

                # Use original tensor for CAM
                cam = gradcam.generate_cam(img)

                plt.subplot(2, 5, i+1)
                plt.title(f'Original\nPred: {pred_label}, True: {true_label}')
                plt.imshow(img.squeeze().cpu().numpy(), cmap='gray')
                plt.axis('off')

                plt.subplot(2, 5, i+6)
                plt.title('GradCAM Heatmap')
                plt.imshow(cam, cmap='jet', alpha=0.5)
                plt.axis('off')

            plt.tight_layout()
            plt.show()
            break

if __name__ == '__main__':
    train_and_visualize()