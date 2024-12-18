import os
import torch
from flask import Flask, request, jsonify, send_file
from PIL import Image
import torchvision.transforms as transforms
from io import BytesIO
from moviepy.editor import VideoFileClip
import numpy as np


from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
import base64
import io
import torch
from PIL import Image
import torchvision.transforms as transforms

# Initialize Flask app
app = Flask(__name__)

# Enable CORS for all domains (for development)
CORS(app)

import torch
import torch.nn as nn
import torchvision.models as models

# Define a Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
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
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out

# Reflection Removal Network with ResNet-style Encoder and Decoder
class ReflectionRemovalNet(nn.Module):
    def __init__(self):
        super(ReflectionRemovalNet, self).__init__()

        # Encoder with ResNet Blocks
        self.encoder = nn.Sequential(
            ResidualBlock(3, 64, stride=1),
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 256, stride=2)
        )
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            ResidualBlock(256, 256, stride=1)
        )
        
        # Decoder with ResNet Blocks
        self.decoder = nn.Sequential(
            ResidualBlock(256, 128, stride=1),
            nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            ResidualBlock(128, 64, stride=1),
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()  # Normalize output to [0, 1]
        )
    
    def forward(self, x):
        enc = self.encoder(x)
        bottleneck = self.bottleneck(enc)
        dec = self.decoder(bottleneck)
        return dec


# Load the model
model = ReflectionRemovalNet()
model.load_state_dict(torch.load(r'D:\MANIPAL\Research\website\anti_reflection\models\new_model_final.pth', map_location=torch.device('cpu')))
model.eval()
import base64
import io
import torch
from flask import Flask, request, jsonify
from PIL import Image
import torchvision.transforms as transforms


# Define any necessary image transformations for input processing
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

@app.route('/process', methods=['POST'])
def process_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Read and process the image
    image = Image.open(file.stream)
    image = transform(image).unsqueeze(0)  # Add batch dimension

    # Forward pass through the model
    with torch.no_grad():
        output = model(image)

    # Convert the output tensor to an image (convert to numpy and then to PIL image)
    output_image = output.squeeze().cpu().numpy()
    output_image = (output_image * 255).astype('uint8')  # Convert to 0-255 range
    output_image = Image.fromarray(output_image.transpose(1, 2, 0))  # Convert to PIL Image

    # Convert image to base64 for sending back to frontend
    buffered = io.BytesIO()
    output_image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    print("done")
    # Return image as base64 string in the response
    return jsonify({'message': 'Image processed successfully', 'image_data': img_str})

if __name__ == '__main__':
    app.run(debug=True)
