import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

# Define the Siamese network class
class Siamese(nn.Module):
    def __init__(self):
        super(Siamese, self).__init__()
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
        self.linear = nn.Sequential(
            nn.Linear(384 * 6 * 5, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.Sigmoid()
        )
        self.out = nn.Sequential(
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward_one(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x

    def forward(self, x1, x2):
        out1 = self.forward_one(x1)
        out2 = self.forward_one(x2)
        batch_size1 = out1.size(0)
        batch_size2 = out2.size(0)
        if batch_size1 > batch_size2:
            pad_size = batch_size1 - batch_size2
            pad_tensor = torch.zeros(pad_size, out2.size(1), out2.size(2), out2.size(3)).to(x2.device)
            out2 = torch.cat((out2, pad_tensor), dim=0)
        elif batch_size1 < batch_size2:
            out2 = out2[:batch_size1]
        concatenated = torch.cat((out1, out2), dim=1)
        dis = torch.abs(out1 - out2)
        out = self.out(dis)
        return out

# Define preprocessing transformations
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((96, 103)),  # Resize images to 96x103
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# Load the trained Siamese network
net = Siamese()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net.to(device)
net.load_state_dict(torch.load("best_model.pt", map_location=device))
net.eval()

# Function to check similarity between two images
def check_similarity(image_path1, image_path2):
    # Load and preprocess images
    image1 = cv2.imread(image_path1)
    image2 = cv2.imread(image_path2)
    image1 = preprocess(image1).to(device)
    image2 = preprocess(image2).to(device)

    # Add batch dimension
    image1 = image1.unsqueeze(0)
    image2 = image2.unsqueeze(0)

    # Forward pass through the Siamese network
    with torch.no_grad():
        output = net(image1, image2)

    # Similarity score
    similarity_score = output.item()  # Assuming the output is a single similarity score

    return similarity_score

# Example usage
image_path1 = "suraj.BMP"
image_path2 = "suraj.BMP"
similarity_score = check_similarity(image_path1, image_path2)
print("Similarity Score:", similarity_score)

# if similarity_score>0.12752:
#     print("Same Finger Print")
# else:
#     print("Different Finger Print")
