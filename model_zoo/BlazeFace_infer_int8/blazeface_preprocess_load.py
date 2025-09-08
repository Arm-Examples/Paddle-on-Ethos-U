import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import sys
import cv2

def preprocess_image(image_path):
    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),          # Resize
        transforms.ToTensor()                   # Convert to Tensor
    ])

    image = Image.open(image_path)
    input_tensor = preprocess(image)
    return input_tensor.unsqueeze(0)          # Add batch dimension

# Check if an image path is provided as a command-line argument
img = sys.argv[1]
image = Image.open(img)
transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor()
])
tensor = transform(image)
print(tensor.shape)  # torch.Size([1, 256, 256, 3])
tensor = tensor.permute(1, 2, 0).unsqueeze(0)  # [C, H, W] to [H, W, C]
print(tensor.shape)  # torch.Size([1, 256, 256, 3])

tensor = tensor * 2 - 1
print(tensor.shape)  # torch.Size([1, 256, 256, 3])
print(tensor.min(), tensor.max())  # tensor(-1., 1.)


quant_scale = 0.007851783186197281
quantized_tensor = tensor / quant_scale
quantized_tensor = torch.clamp(quantized_tensor, -127, 127)  # Ensure within int8 range
int8_tensor = quantized_tensor.to(torch.int8)
print(int8_tensor.shape)  #  torch.Size([1, 256, 256, 3])
print(int8_tensor.min(), int8_tensor.max())  # tensor(-1., 1.)
with open("verify/input_tensor.bin", "wb") as f:
    f.write(int8_tensor.numpy().tobytes())
'''
groundtruth_out = sys.argv[2]
bench_data = np.load(groundtruth_out)
bench_data_transposed = np.transpose(bench_data, (0, 2, 3, 1))
print(bench_data_transposed.shape)
print(bench_data_transposed.min(), bench_data_transposed.max())  # tensor(-1., 1.)
output_file = 'verify/input_tensor2.bin'
bench_data_transposed.tofile(output_file)
'''
