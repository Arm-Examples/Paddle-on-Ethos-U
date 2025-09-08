import argparse 
import numpy as np
import re

def hex_to_signed_int8(hex_str):
    '''convert hex string to signed int8'''
    num = int(hex_str, 16)
    return num - 256 if num > 127 else num

def dequantize(hex_values, scale, zero_point):
    '''dequantize'''
    return [scale * (hex_to_signed_int8(h) - zero_point) for h in hex_values]

def load_labels(label_file):
    '''load labels'''
    with open(label_file, 'r') as f:
        return [line.strip() for line in f.readlines()]

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def parse_output_file(output_file):
    '''Get hex values from output file'''
    hex_values = []
    with open(output_file, 'r') as f:
        for line in f:
            match = re.search(r'paddle lite arm output:\s*([0-9a-fA-F]{1,2})', line)
            if match:
                hex_values.append(match.group(1))
    return hex_values

def main():
    parser = argparse.ArgumentParser(description='dequantization tool for model output')
    parser.add_argument('input_file', help='output file path (hexadecimal data)')
    parser.add_argument('label_file', help='ImageNet label file path')
    args = parser.parse_args()

    # Quantization parameters (modify according to actual model configuration)
    scale = 0.035428     # scale factor
    zero_point = 190

    # load hex values from input file
    hex_values = parse_output_file(args.input_file)

    result = dequantize(hex_values, scale, zero_point)

    # load labels
    labels = load_labels(args.label_file)

    # calculate probabilities
    probabilities = softmax(np.array(result))

    # print out top 5 results
    top5_indices = np.argsort(probabilities)[-5:][::-1]
    print("\nTop 5 prediction results:")
    for idx in top5_indices:
        print(f"{labels[idx]:<30} {probabilities[idx]*100:.2f}%")

if __name__ == "__main__":
    main()
