import numpy as np
import sys
import cv2

def preprocess_image(image_path):
    image = cv2.imread(image_path)

    # Preprocessing float32 data.
    image = image.astype(np.float32) / 255.0
    normalized_img = (image - 0.5) / 0.5
    nhwc_img = np.expand_dims(normalized_img, axis=0)  # add batch dimension

    return nhwc_img

img = sys.argv[1]
image = preprocess_image(img)

quant_scale = 0.007874015718698502
quantized_tensor = image / quant_scale
print(quantized_tensor.shape, quantized_tensor.dtype)
print(quantized_tensor.min(), quantized_tensor.max())

with open("verify/input_tensor.bin", "wb") as f:
    f.write(quantized_tensor.astype(np.int8))
