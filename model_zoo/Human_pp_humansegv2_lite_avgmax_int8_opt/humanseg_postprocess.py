import torch
import numpy as np
import cv2
import torch.nn.functional as F
import sys
import os

def load_data(bin_file, shape, dtype):
    data = np.fromfile(bin_file, dtype=dtype)
    data = data.reshape(shape)
    return data

def load_image(image_path):
    raw_image = cv2.imread(image_path)  # BGR
    image = cv2.resize(raw_image, (512, 512), interpolation=cv2.INTER_AREA)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert to RGB
    return image, raw_image.shape[0], raw_image.shape[1]

def highlight_person(image, mask):
    # create a yellow mask
    yellow_mask = np.zeros_like(image, dtype=np.uint8)
    yellow_mask[mask == 1] = [255, 255, 0]  # Yellow (RGB)

    highlighted_image = cv2.addWeighted(image, 0.7, yellow_mask, 0.3, 0)
    return highlighted_image

# ========== 1. Load Image ==========
image_path = sys.argv[1]
image, raw_height, raw_width = load_image(image_path)

# ========== 2. logits ==========
# Assume the model output is 2 channels (background & person)
data_path = sys.argv[2]
running_data = load_data(data_path, [512,512,2], np.int8)
logits = running_data.transpose((2,0,1)) * 0.033177462149792766

# ========== 3. Get Face Mask ==========
mask = torch.argmax(torch.from_numpy(logits), dim=0).squeeze(0).cpu().numpy()  # (512, 512)

# ========== 4. Blend Mask ==========
# Blend yellow mask with the original image
highlighted_image = highlight_person(image, mask)

# ========== 5. Display Result ==========
def show_images(image, highlighted_image):
    # Create a concatenated image (horizontal concatenation)

    resized_highlighted_image = cv2.resize(highlighted_image, (raw_width, raw_height))

    result_folder = os.path.join(*(image_path.split('/')[:-1]))
    result_image_path = f"{result_folder}/seg_output_{image_path.split('/')[-1]}"
    print("Result saved to:", result_image_path)
    cv2.imwrite(result_image_path, resized_highlighted_image)

    # Display the images
    # adjust the size of the combined image to fit the screen
    #
    # combined_image = np.hstack((image, highlighted_image))
    # resized_image = cv2.resize(combined_image, (raw_width*2, raw_height))
    # cv2.imshow("Original (Left) | Highlighted (Right)", resized_image)
    # while True:
    #     if cv2.waitKey(20) & 0xFF == 27:  # press `ESC` to exit
    #         break
    #     if cv2.getWindowProperty("Original (Left) | Highlighted (Right)", cv2.WND_PROP_VISIBLE) < 1:
    #         break
    #
    # cv2.destroyAllWindows()

# BGR to RGB
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
highlighted_image = cv2.cvtColor(highlighted_image, cv2.COLOR_BGR2RGB)

show_images(image, highlighted_image)
