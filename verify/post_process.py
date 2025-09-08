import argparse 
import numpy as np
import re

def hex_to_signed_int8(hex_str):
    """Convert hexadecimal string to signed int8"""
    num = int(hex_str, 16)
    return num - 256 if num > 127 else num

def dequantize(hex_values, scale, zero_point):
    """Perform dequantization"""
    return [scale * (hex_to_signed_int8(h) - zero_point) for h in hex_values]

def load_labels(label_file):
    """Load labels from file"""
    with open(label_file, 'r') as f:
        return [line.strip() for line in f.readlines()]

def softmax(x):
    """Calculate softmax"""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def parse_c_array(c_array_str):
    """Parse C-style array and extract hexadecimal values"""
    # Use regular expression to match hexadecimal numbers
    hex_values = re.findall(r'0[xX]([0-9a-fA-F]{1,2})', c_array_str)
    return hex_values

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Model output dequantization tool')
    parser.add_argument('input_file', help='C array file path (hexadecimal data)')
    parser.add_argument('label_file', help='ImageNet label file path')
    args = parser.parse_args()

    # Quantization parameters (modify according to actual model configuration)
    scale = 0.035428     # Scale factor
    zero_point = 190     # Zero point

    # Read hexadecimal numbers from input file
    with open(args.input_file, 'r') as infile:
        c_array_str = infile.read()
    
    # Parse hexadecimal values
    hex_values = parse_c_array(c_array_str)

    # Dequantization calculation
    result = dequantize(hex_values, scale, zero_point)

    # Load labels
    labels = load_labels(args.label_file)

    # Calculate probabilities
    probabilities = softmax(np.array(result))

    # Get and print TOP5 results
    top5_indices = np.argsort(probabilities)[-5:][::-1]
    print("\nTop 5 prediction results:")
    for idx in top5_indices:
        print(f"{labels[idx]:<30} {probabilities[idx]*100:.2f}%")

if __name__ == "__main__":
    main()

