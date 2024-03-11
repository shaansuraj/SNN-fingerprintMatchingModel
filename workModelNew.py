# PyTorch libraries and modules
from sklearn.model_selection import train_test_split
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.functional import pairwise_distance
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import os
import cv2
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, roc_auc_score

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the Siamese network class
class Siamese(nn.Module):
    def __init__(self, input_height, input_width):
        super(Siamese, self).__init__()
        self.input_height = input_height
        self.input_width = input_width 
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, (10, 10), stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, (7, 7), stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, (5, 5), stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 384, (3, 3)),
            nn.ReLU(inplace=True),
        )
        
        # Calculate the flattened size dynamically based on input size
         # Calculate the flattened size
        output_height = input_height
        output_width = input_width
        for layer in self.conv:
            if isinstance(layer, nn.Conv2d):
                output_height = (output_height - layer.kernel_size[0] + 1) // 2
                output_width = (output_width - layer.kernel_size[1] + 1) // 2

        self.flattened_size = output_height * output_width * 384

        # Linear layers
        self.linear = nn.Sequential(
            nn.Linear(384 * self.flattened_size, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.Sigmoid()
        )

    def forward_one(self, x):
        x = self.conv(x)
        if self.flattened_size is None:
            self.flattened_size = x.view(x.size(0), -1).size(1)  # Calculate flattened size
            self.linear[0] = nn.Linear(384 * self.flattened_size, 2048)  # Update linear layer based on calculated size
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x

    def forward(self, x1, x2):
        x1, x2 = x1.to(device), x2.to(device)
        out1 = self.forward_one(x1)
        out2 = self.forward_one(x2)
        
        # Calculate the Euclidean distance between embeddings
        distance = pairwise_distance(out1, out2)
        
        return distance

# Define a custom dataset class to load processed images and labels
class CustomDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        # Convert image array to uint8
        image = (image * 255).astype(np.uint8)
        
        # Convert RGBA image to RGB
        image = Image.fromarray(image)
        image = image.convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        return image, label

# Define preprocessing transformations
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# Load processed images and labels
processed_images_dir = "processedThumbs"
labels_file = os.path.join('labelArrays', 'processedThumbs.npy')
# Load processed images array
processed_images_array = np.load(labels_file)
images = []
for root, dirs, files in os.walk(processed_images_dir):
    for file in files:
        image_path = os.path.join(root, file)
        image = cv2.imread(image_path)
        images.append(image)

# Extract numerical prefixes from image filenames to use as labels
labels = [int(os.path.basename(filename).split('_')[0]) for filename in os.listdir(processed_images_dir)]

# Split data into train and test sets
train_images, test_images, train_labels, test_labels = train_test_split(processed_images_array, labels, test_size=0.2, random_state=42)

# Create dataset instances
train_dataset = CustomDataset(train_images, train_labels, transform=preprocess)
test_dataset = CustomDataset(test_images, test_labels, transform=preprocess)

# Define DataLoader
batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Instantiate the Siamese network
input_height = 103
input_width = 96
net = Siamese(input_height=input_height, input_width=input_width).to(device) 
# Define contrastive loss
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, distance, label):
        loss_contrastive = torch.mean((1 - label) * torch.pow(distance, 2) +
                                       (label) * torch.pow(F.relu(self.margin - distance), 2))
        return loss_contrastive

criterion = ContrastiveLoss()

# Define optimizer
optimizer = optim.Adam(net.parameters(), lr=0.001)

# Learning rate scheduler
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# Define early stopping parameters
patience = 3
early_stop_counter = 0
best_auc = 0.0

# Training loop
n_epochs = 50
for epoch in range(n_epochs):
    print(f"Epoch [{epoch+1}/{n_epochs}]")
    net.train()
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        images, labels = data  # Unpack images and labels
        labels = torch.zeros(images.size(0), 1)  # Generate dummy labels (since we're not training Siamese network)
        images, labels = images.to(device), labels.to(device) 

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = net(images, images)
        loss = criterion(outputs, labels.float())

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Print statistics
        running_loss += loss.item()

    # Print epoch statistics
    avg_loss = running_loss / len(train_loader)
    print(f"  Train Loss: {avg_loss:.4f}")

    # Validation loss
    net.eval()
    true_labels = []
    predictions = []
    with torch.no_grad():
        for data in test_loader:
            images, labels = data  # Unpack images and labels
            labels = torch.zeros(images.size(0), 1)  # Generate dummy labels
            images, labels = images.to(device), labels.to(device) 
            outputs = net(images, images)  # Pass both images to forward method with self-similarity
            predictions.extend(outputs.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    # Calculate AUC
    auc = roc_auc_score(true_labels, predictions)
    print(f"  AUC: {auc:.4f}")

    # Update learning rate scheduler
    scheduler.step()

    # Early stopping
    if auc > best_auc:
        best_auc = auc
        early_stop_counter = 0
        torch.save(net.state_dict(), "best_model.pt")  # Save the best model
    else:
        early_stop_counter += 1
        if early_stop_counter >= patience:
            print("Early stopping triggered!")
            break

# Load the best model
net.load_state_dict(torch.load("best_model.pt"))

# Evaluate on test set
net.eval()
true_labels = []
predictions = []
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        outputs = net(images)
        predictions.extend(outputs.tolist())
        true_labels.extend(labels.tolist())

# Choose an appropriate threshold based on validation performance
threshold = 0.5  # Example threshold

# Calculate MSE
mse = mean_squared_error(true_labels, predictions)
print("Mean Squared Error:", mse)
