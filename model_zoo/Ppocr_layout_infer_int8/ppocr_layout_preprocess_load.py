import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import sys
import cv2

def preprocess_image(image_path):
    preprocess = transforms.Compose([
        transforms.Resize((640, 640)),          # Resize
        transforms.ToTensor()                   # Convert to Tensor
    ])

    image = Image.open(image_path)
    input_tensor = preprocess(image)
    return input_tensor.unsqueeze(0)          # Add batch dimension

# Check if an image path is provided as a command-line argument
img = sys.argv[1]
image = Image.open(img)
image = image.resize((608, 800))
image = np.array(image).transpose(2, 0, 1)
print(image.shape)  # torch.Size([1, 256, 256, 3])
image = image.astype(np.float32) / 255.0
image = np.expand_dims(image, axis=0)
image -= 0.5
image /= 0.5
with open("verify/input_tensor.bin", "wb") as f:
    f.write(image.tobytes())

'''
groundtruth_out = sys.argv[2]
bench_data = np.load(groundtruth_out)
bench_data_transposed = np.transpose(bench_data, (0, 2, 3, 1))
print(bench_data_transposed.shape)  # torch.Size([1, 256, 256, 3])
print(bench_data_transposed.min(), bench_data_transposed.max())  # tensor(-1., 1.)
output_file = 'verify/input_tensor.bin'
    
#bench_data_transposed = bench_data_transposed / 0.06848310679197311
#bench_data_transposed = bench_data_transposed.astype(np.int8)
print(bench_data_transposed.min(), bench_data_transposed.max())  # tensor(-1., 1.)
bench_data_transposed.tofile(output_file)
'''

