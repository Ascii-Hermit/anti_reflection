import os
import torch
from flask import Flask, request, jsonify
from PIL import Image
from io import BytesIO
import torchvision.transforms as transforms

# Initialize the Flask app
app = Flask(__name__)

# Load your trained model (final_model.pth)
import torch

# Load the model on the CPU (if CUDA is not available)
model = torch.load('model_epoch_100.pth', map_location=torch.device('cpu'))

# If you want to transfer the model back to GPU later, you can do it as follows (only if CUDA is available)
if torch.cuda.is_available():
    model = model.to(torch.device('cuda'))

model.eval()  # Set the model to evaluation mode

# Define any necessary image transformations for input processing
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# API route to process the image
@app.route('/process', methods=['POST'])
def process_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Read and process the image
    image = Image.open(BytesIO(file.read()))
    image = transform(image).unsqueeze(0)  # Add batch dimension

    # Forward pass through the model
    with torch.no_grad():
        output = model(image)

    # Convert the output tensor to an image (if necessary)
    result_image = output.squeeze().cpu().numpy()  # Example to convert tensor to numpy

    # You could return the processed result as a base64 encoded image or save it
    return jsonify({'message': 'Image processed successfully'})

if __name__ == '__main__':
    app.run(debug=True)
