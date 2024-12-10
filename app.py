import os
import torch
from flask import Flask, request, jsonify, send_file
from PIL import Image
import torchvision.transforms as transforms
from io import BytesIO
from moviepy.editor import VideoFileClip
import numpy as np


app = Flask(__name__)

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

# Reflection Removal Network with Deeper ResNet-style Encoder and Decoder
class ReflectionRemovalNet(nn.Module):
    def __init__(self):
        super(ReflectionRemovalNet, self).__init__()

        # Encoder with Deeper ResNet Blocks (No Attention)
        self.encoder = nn.Sequential(
            ResidualBlock(3, 64, stride=1),
            ResidualBlock(64, 64, stride=1),  # Additional Residual Block
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 128, stride=1),  # Additional Residual Block
            ResidualBlock(128, 256, stride=2),
            ResidualBlock(256, 256, stride=1),  # Additional Residual Block
            ResidualBlock(256, 512, stride=2),  # Deeper block with more filters
        )
        
        # Bottleneck with Residual Block
        self.bottleneck = nn.Sequential(
            ResidualBlock(512, 512, stride=1),
            ResidualBlock(512, 512, stride=1),  # Additional Residual Block
        )
        
        # Decoder with Deeper ResNet Blocks (No Attention)
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
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()  # Normalize output to [0, 1]
        )
    
    def forward(self, x):
        enc = self.encoder(x)
        bottleneck = self.bottleneck(enc)
        dec = self.decoder(bottleneck)
        return dec

class PerceptualLoss(nn.Module):
    def __init__(self, vgg19_model_path=r"/kaggle/input/vg19/pytorch/default/1/vgg19-dcbb9e9d.pth"):
        super(PerceptualLoss, self).__init__()
        if vgg19_model_path:
            # Initialize a VGG19 model and load the state_dict
            vgg = models.vgg19(pretrained=False)
            vgg.load_state_dict(torch.load(vgg19_model_path))  # Load weights into the VGG model
            vgg = vgg.features
        else:
            vgg = models.vgg19(pretrained=True).features  # Load the pretrained VGG model from torchvision
        
        self.layers = nn.Sequential(*list(vgg[:16])).eval()  # Use layers up to relu4_1
        for param in self.layers.parameters():
            param.requires_grad = False  # Freeze VGG weights

    def forward(self, input, target):
        input_features = self.layers(input)
        target_features = self.layers(target)
        return nn.functional.l1_loss(input_features, target_features)

# Reconstruction Loss for Pixel-Level Accuracy
class ReconstructionLoss(nn.Module):
    def __init__(self, loss_type='l1'):
        """
        Initializes the reconstruction loss.
        Args:
            loss_type (str): Type of pixel-level loss ('l1' or 'l2').
        """
        super(ReconstructionLoss, self).__init__()
        if loss_type == 'l1':
            self.loss_fn = nn.L1Loss()
        elif loss_type == 'l2':
            self.loss_fn = nn.MSELoss()
        else:
            raise ValueError("loss_type must be 'l1' or 'l2'")

    def forward(self, predicted, target):
        """
        Computes the reconstruction loss.
        Args:
            predicted (Tensor): The reconstructed image from the model.
            target (Tensor): The ground truth target image.
        Returns:
            Tensor: Computed reconstruction loss.
        """
        return self.loss_fn(predicted, target)


# Load the model
model = ReflectionRemovalNet()
model.load_state_dict(torch.load('model_epoch_100.pth', map_location=torch.device('cpu')))
model.eval()

# Transformations for input images
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# API to process the image or video
@app.route('/process', methods=['POST'])
def process_media():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Check if the file is an image or video
    if file.content_type.startswith('image'):
        return process_image(file)
    elif file.content_type.startswith('video'):
        return process_video(file)
    else:
        return jsonify({'error': 'Unsupported file type'}), 400

def process_image(file):
    # Process the image
    image = Image.open(file)
    image = transform(image).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        output = model(image)

    # Convert the output tensor to an image
    output_image = output.squeeze().cpu().numpy()  # Convert tensor to numpy array
    output_image = (output_image * 255).astype(np.uint8)  # Scale back to [0, 255]
    output_image = Image.fromarray(output_image.transpose(1, 2, 0))  # Convert channels to RGB

    # Save the processed image
    output_image_path = 'processed_image.jpg'
    output_image.save(output_image_path)

    return jsonify({
        'message': 'Image processed successfully',
        'processed_image_url': output_image_path  # URL or base64 string can be used
    })

def process_video(file):
    # Save the video file temporarily
    video_path = 'input_video.mp4'
    file.save(video_path)

    # Process video frame by frame
    processed_video_path = 'processed_video.mp4'
    video_clip = VideoFileClip(video_path)
    
    def process_frame(frame):
        # Convert the frame to an image and process it through the model
        image = Image.fromarray(frame)
        image = transform(image).unsqueeze(0)

        with torch.no_grad():
            output = model(image)

        # Convert the output tensor to an image
        output_image = output.squeeze().cpu().numpy()
        output_image = (output_image * 255).astype(np.uint8)
        output_frame = output_image.transpose(1, 2, 0)
        return output_frame

    # Apply the processing function to each frame and write to a new video
    processed_clip = video_clip.fl_image(process_frame)
    processed_clip.write_videofile(processed_video_path, codec='libx264')

    return jsonify({
        'message': 'Video processed successfully',
        'processed_video_url': processed_video_path  # Return the path to processed video
    })

if __name__ == '__main__':
    app.run(debug=True)
