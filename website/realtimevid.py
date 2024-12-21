import cv2
import torch
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt

# Model imports and setup
import torch.nn as nn
import torchvision.models as models

# Define ResidualBlock with Dropout
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dropout_prob=0.3):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(p=dropout_prob)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.skip_connection = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip_connection = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = self.skip_connection(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out

class ReflectionRemovalNet(nn.Module):
    def __init__(self):
        super(ReflectionRemovalNet, self).__init__()

        # Updated encoder with one more ResidualBlock
        self.encoder = nn.Sequential(
            ResidualBlock(3, 64, stride=1),
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 256, stride=2),
            ResidualBlock(256, 512, stride=2),
            ResidualBlock(512, 512, stride=2),  # New layer added here
        )

        self.bottleneck = nn.Sequential(
            ResidualBlock(512, 512, stride=1)
        )

        # Updated decoder with one more ResidualBlock
        self.decoder = nn.Sequential(
            ResidualBlock(512, 256, stride=1),
            nn.ConvTranspose2d(256, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            ResidualBlock(256, 128, stride=1),
            nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            ResidualBlock(128, 64, stride=1),
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            ResidualBlock(64, 64, stride=1),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)  # Added layer
        )

    def forward(self, x):
        enc = self.encoder(x)
        bottleneck = self.bottleneck(enc)
        dec = self.decoder(bottleneck)
        return dec

# Check if CUDA (GPU) is available and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the model and move it to the appropriate device (GPU or CPU)
model = ReflectionRemovalNet().to(device)
model.load_state_dict(torch.load(r'D:\MANIPAL\Research\website\anti_reflection\models\model_epoch_30.pth', map_location=device))
model.eval()

# Define a transform to preprocess the video frames
transform = transforms.Compose([
    transforms.ToPILImage(),          # Convert numpy array to PIL image
    transforms.Resize((224, 224)),    # Resize to the expected input size for the model (adjust based on your model)
    transforms.ToTensor(),            # Convert to tensor
])

# Open the webcam
cap = cv2.VideoCapture(0)

# Get the original frame size
ret, frame = cap.read()
if not ret:
    print("Failed to capture video frame")
    cap.release()
    exit()

# Get the height and width of the original frame
frame_height, frame_width = frame.shape[:2]

# Create a figure for displaying normal and processed video side by side
plt.ion()  # Turn on interactive mode
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))  # Two subplots for side by side

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # Convert the frame to RGB (OpenCV uses BGR by default)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Apply preprocessing transformations to the frame
    input_tensor = transform(frame_rgb)
    input_tensor = input_tensor.unsqueeze(0).to(device)  # Add batch dimension and move tensor to GPU/CPU
    
    # Run the model to get predictions
    with torch.no_grad():  # Disable gradient calculation for inference
        output = model(input_tensor)  # Process the frame through the model

    # If your model outputs an image (e.g., for reflection removal), convert it back to a displayable format
    output_img = output.squeeze(0).permute(1, 2, 0).cpu().numpy()  # Convert tensor to numpy array and move it to CPU
    output_img = np.clip(output_img, 0, 1)  # Clip the values to the valid range [0, 1]
    output_img = (output_img * 255).astype(np.uint8)  # Convert to uint8 format for display
    
    # Resize the processed output image to match the original input frame size
    output_img_resized = cv2.resize(output_img, (frame_width, frame_height))

    # Display the normal and processed frames side by side using matplotlib
    ax1.imshow(frame_rgb)  # Show normal (original) frame
    ax1.axis('off')  # Turn off the axis
    ax1.set_title("Normal Video")

    ax2.imshow(output_img_resized)  # Show processed (output) frame
    ax2.axis('off')  # Turn off the axis
    ax2.set_title("Processed Video")

    plt.draw()  # Update the plot
    plt.pause(0.01)  # Allow the frame to update (this is crucial for real-time)

    # Exit condition for interactive mode
    if plt.waitforbuttonpress(timeout=0.01):  # Detect a keypress
        break

# Release the webcam and close the matplotlib window
cap.release()
plt.close(fig)  # Close the matplotlib window
