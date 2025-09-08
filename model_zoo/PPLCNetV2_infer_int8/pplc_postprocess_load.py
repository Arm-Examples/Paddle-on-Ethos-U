import numpy as np
import sys
import ast
import torch
import torch.nn.functional as F

def load_data(bin_file, shape, dtype):
    data = np.fromfile(bin_file, dtype=dtype)
    data = data.reshape(shape)
    return data

def post_process(input_data):

    input_tensor = torch.from_numpy(input_data)
    softmax_out = input_tensor.view(input_tensor.size(0), -1)
    top5_probs, top5_indices = torch.topk(softmax_out, k=5, dim=1)
    with open('./model_zoo/PPLCNetV2_infer_int8/imagenet_labels.txt', 'r') as f:
        labels = f.readlines()
        labels = [label.strip() for label in labels]
    print("Top 5 Classification Results:")
    for i in range(5):
        index = top5_indices[0][i].item()
        prob = top5_probs[0][i].item()
        label = labels[index]
        print(f"Rank {i + 1}: {label} (Probability: {prob:.4f})")

if __name__ == "__main__":
    test_shape = list(map(int, sys.argv[2].split(",")))
    running_out = sys.argv[1]

    # fp32 
    running_data = load_data(running_out, test_shape, "<f4")
    input_data = np.array(running_data).astype(np.float32)
    post_process(input_data)
    
