import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import sys

def preprocess_image(image_path):
    preprocess = transforms.Compose([
        transforms.Resize((512, 512)),          # Resize
        transforms.ToTensor(),                   # Convert to Tensor
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],         # ImageNet standardization parameters
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    image = Image.open(image_path)
    input_tensor = preprocess(image)
    return input_tensor.unsqueeze(0)          # Add batch dimension

# Check if an image path is provided as a command-line argument
if len(sys.argv) != 2:
    print("Usage: python humanseg_preprocess_load.py <image_path>")
    sys.exit(1)

image_path = sys.argv[1]

# Preprocess the image
input_batch = preprocess_image(image_path)
nhwc_tensor = input_batch.permute(0, 2, 3, 1)  # Convert from NCHW to NHWC

# Quantization
quant_scale = 0.02016254886984825
quantized_tensor = nhwc_tensor / quant_scale  # Scale
quantized_tensor = torch.clamp(quantized_tensor, -128, 127)  # Ensure within int8 range
int8_tensor = quantized_tensor.to(torch.int8)

# Save the tensor directly as int8 (binary format)
with open("verify/input_tensor.bin", "wb") as f:
    f.write(int8_tensor.numpy().tobytes())

