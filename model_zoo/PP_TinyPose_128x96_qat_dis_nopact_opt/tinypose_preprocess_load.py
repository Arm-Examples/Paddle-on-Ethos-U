import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import sys

def preprocess_image(image_path):
    preprocess = transforms.Compose([
        transforms.Resize((128, 96)),  # Change image size
        transforms.ToTensor(),  # Convert to Tensor
        # Normalize the image with mean and std
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Open the image
    image = Image.open(image_path).convert("RGB")
    # Preprocess the image
    input_tensor = preprocess(image)
    # Add batch dimension
    return input_tensor.unsqueeze(0)

# Check if an image path is provided as a command-line argument
if len(sys.argv) != 2:
    print("Usage: python tinypose_preprocess_load.py <image_path>")
    sys.exit(1)

# Get Image path from command line argument
image_path = sys.argv[1]

input_batch = preprocess_image(image_path)
# Convert NCHW to NHWC
nhwc_tensor = input_batch.permute(0, 2, 3, 1)

# Quantize the tensor
quant_scale = 0.02016254886984825
quantized_tensor = nhwc_tensor / quant_scale
# Conform to int8 range
quantized_tensor = torch.clamp(quantized_tensor, -128, 127)
# Convert to int8 type
int8_tensor = quantized_tensor.to(torch.int8)

# Write tensor to binary file
with open("./verify/input_tensor.bin", "wb") as f:
    f.write(int8_tensor.numpy().tobytes())
