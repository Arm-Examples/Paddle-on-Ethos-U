import torch
import torch.nn as nn
import numpy as np
import argparse


class LSTMImplementation(nn.Module):
    def __init__(self, hidden_size=64, bidirectional=True, num_layers=1):
        super(LSTMImplementation, self).__init__()
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.num_layers = num_layers

    def forward(
        self,
        X,
        W,
        R,
        B=None,
        sequence_lens=None,
        initial_h=None,
        initial_c=None,
        P=None,
    ):
        seq_length, batch_size, input_size = X.shape

        if initial_h is None:
            initial_h = torch.zeros(
                self.num_directions, batch_size, self.hidden_size, device=X.device
            )
        if initial_c is None:
            initial_c = torch.zeros(
                self.num_directions, batch_size, self.hidden_size, device=X.device
            )

        if B is not None:

            W_bias = B[:, : 4 * self.hidden_size]
            R_bias = B[:, 4 * self.hidden_size : 8 * self.hidden_size]
        else:
            W_bias = torch.zeros(
                self.num_directions, 4 * self.hidden_size, device=X.device
            )
            R_bias = torch.zeros(
                self.num_directions, 4 * self.hidden_size, device=X.device
            )

        if P is None:
            P = torch.zeros(self.num_directions, 3 * self.hidden_size, device=X.device)

        Y = torch.zeros(
            seq_length,
            self.num_directions,
            batch_size,
            self.hidden_size,
            device=X.device,
        )
        Y_h = torch.zeros(
            self.num_directions, batch_size, self.hidden_size, device=X.device
        )
        Y_c = torch.zeros(
            self.num_directions, batch_size, self.hidden_size, device=X.device
        )

        for direction in range(self.num_directions):
            h_t = initial_h[direction]
            c_t = initial_c[direction]

            seq_indices = range(seq_length)
            if direction == 1 and self.bidirectional:
                seq_indices = reversed(list(seq_indices))

            W_dir = W[direction]
            R_dir = R[direction]
            W_bias_dir = W_bias[direction]
            R_bias_dir = R_bias[direction]

            if P is not None:
                P_dir = P[direction]
                P_i = P_dir[: self.hidden_size]
                P_f = P_dir[self.hidden_size : 2 * self.hidden_size]
                P_o = P_dir[2 * self.hidden_size : 3 * self.hidden_size]

            for t in seq_indices:
                x_t = X[t]

                W_i = W_dir[: self.hidden_size]
                W_o = W_dir[self.hidden_size : 2 * self.hidden_size]
                W_f = W_dir[2 * self.hidden_size : 3 * self.hidden_size]
                W_c = W_dir[3 * self.hidden_size : 4 * self.hidden_size]

                R_i = R_dir[: self.hidden_size]
                R_o = R_dir[self.hidden_size : 2 * self.hidden_size]
                R_f = R_dir[2 * self.hidden_size : 3 * self.hidden_size]
                R_c = R_dir[3 * self.hidden_size : 4 * self.hidden_size]

                Wb_i = W_bias_dir[: self.hidden_size]
                Wb_o = W_bias_dir[self.hidden_size : 2 * self.hidden_size]
                Wb_f = W_bias_dir[2 * self.hidden_size : 3 * self.hidden_size]
                Wb_c = W_bias_dir[3 * self.hidden_size : 4 * self.hidden_size]

                Rb_i = R_bias_dir[: self.hidden_size]
                Rb_o = R_bias_dir[self.hidden_size : 2 * self.hidden_size]
                Rb_f = R_bias_dir[2 * self.hidden_size : 3 * self.hidden_size]
                Rb_c = R_bias_dir[3 * self.hidden_size : 4 * self.hidden_size]

                i_t = torch.matmul(x_t, W_i.t()) + Wb_i
                i_t = i_t + torch.matmul(h_t, R_i.t()) + Rb_i
                if P is not None:
                    i_t = i_t + P_i * c_t
                i_t = torch.sigmoid(i_t)

                f_t = torch.matmul(x_t, W_f.t()) + Wb_f
                f_t = f_t + torch.matmul(h_t, R_f.t()) + Rb_f
                if P is not None:
                    f_t = f_t + P_f * c_t
                f_t = torch.sigmoid(f_t)

                c_tilde = torch.matmul(x_t, W_c.t()) + Wb_c
                c_tilde = c_tilde + torch.matmul(h_t, R_c.t()) + Rb_c
                c_tilde = torch.tanh(c_tilde)

                c_t = f_t * c_t + i_t * c_tilde

                o_t = torch.matmul(x_t, W_o.t()) + Wb_o
                o_t = o_t + torch.matmul(h_t, R_o.t()) + Rb_o
                if P is not None:
                    o_t = o_t + P_o * c_t
                o_t = torch.sigmoid(o_t)

                h_t = o_t * torch.tanh(c_t)

                if sequence_lens is not None:
                    mask = (t < sequence_lens).float().unsqueeze(1)
                    h_t = h_t * mask + initial_h[direction] * (1 - mask)
                    c_t = c_t * mask + initial_c[direction] * (1 - mask)

                Y[t, direction] = h_t

            Y_h[direction] = h_t
            Y_c[direction] = c_t

        return Y, Y_h, Y_c


def conv2realoutput(conv_output, weight_path):
    """
    Convert convolution output to final recognition result probabilities

    Args:
        conv_output: Output of the convolution layer

    Returns:
        Probability distribution of recognition results
    """
    lstm = LSTMImplementation(hidden_size=64, bidirectional=True)

    x = torch.nn.functional.hardswish(conv_output)
    x = torch.nn.functional.max_pool2d(
        x, kernel_size=(2, 2), stride=(2, 2), padding=(0, 0)
    )
    x = x.squeeze(2).permute(2, 0, 1)

    # First LSTM layer
    W = np.load(f"{weight_path}/w1.npy")
    R = np.load(f"{weight_path}/r1.npy")
    B = np.load(f"{weight_path}/b1.npy")
    W = torch.tensor(W, dtype=torch.float32)
    R = torch.tensor(R, dtype=torch.float32)
    B = torch.tensor(B, dtype=torch.float32)
    initial_h = torch.zeros(2, 1, 64)
    initial_c = torch.zeros(2, 1, 64)

    x, Y_h, Y_c = lstm(x, W, R, B, initial_h=initial_h, initial_c=initial_c)

    x = x.permute(0, 2, 1, 3).reshape(320, 1, 128)

    # Second LSTM layer
    W = np.load(f"{weight_path}/w2.npy")
    R = np.load(f"{weight_path}/r2.npy")
    B = np.load(f"{weight_path}/b2.npy")
    W = torch.tensor(W, dtype=torch.float32)
    R = torch.tensor(R, dtype=torch.float32)
    B = torch.tensor(B, dtype=torch.float32)
    initial_h = torch.zeros(2, 1, 64)
    initial_c = torch.zeros(2, 1, 64)

    x, Y_h, Y_c = lstm(x, W, R, B, initial_h=initial_h, initial_c=initial_c)

    x = x.permute(0, 2, 1, 3).reshape(320, 1, 128).permute(1, 0, 2)

    # First fully connected layer
    w1 = np.load(f"{weight_path}/linear1_w.npy")
    b1 = np.load(f"{weight_path}/linear1_b.npy")
    w1 = torch.tensor(w1, dtype=torch.float32)
    b1 = torch.tensor(b1, dtype=torch.float32)

    x = torch.matmul(x, w1) + b1

    # Second fully connected layer
    w2 = np.load(f"{weight_path}/linear2_w.npy")
    b2 = np.load(f"{weight_path}/linear2_b.npy")
    w2 = torch.tensor(w2, dtype=torch.float32)
    b2 = torch.tensor(b2, dtype=torch.float32)

    x = torch.matmul(x, w2) + b2

    x = torch.softmax(x, dim=-1)

    return x


def dequantize(input_data, scale):
    """
    Dequantize the quantized data

    Args:
        input_data: Quantized data
        scale: Quantization scale factor

    Returns:
        Dequantized floating-point data
    """
    return torch.tensor(input_data.astype(np.float32) * scale)


def load_character_dict(dict_path):
    """
    Load character dictionary

    Args:
        dict_path: Path to the dictionary file, one character per line

    Returns:
        List of characters, including all characters
    """
    characters = []
    with open(dict_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                characters.append(line)

    # Ensure the dictionary contains a blank character for CTC decoding
    characters.insert(0, "#")  # '#' as the CTC blank token

    if " " not in characters:  # Add space character
        characters.append(" ")
    characters.append("")

    return characters


def ctc_decode(probs, characters):
    """
    Simple CTC decoding

    Args:
        probs: Model output probabilities, shape [seq_len, num_classes]
        characters: List of characters

    Returns:
        Decoded text and confidence score
    """
    # Check input
    if len(probs.shape) != 2:
        raise ValueError(f"Expected shape [seq_len, num_classes], But got {probs.shape}")

    # Find the index of the maximum probability at each time step
    pred_indices = np.argmax(probs, axis=1)
    # print(f"Predicted index sequence: {pred_indices}")

    # Store characters and scores
    text = ""
    char_scores = []
    last_index = 0 # Used to remove repeated characters

    # Iterate through predictions at each time step
    for idx, index in enumerate(pred_indices):
        # If the index is greater than 0 (not CTC blank) and not equal to the previous index (remove repeated characters)
        if index > 0 and (index != last_index or last_index == 0):
            if index < len(characters):  # Ensure the index is valid
                text += characters[index]
                char_scores.append(probs[idx, index])
            else:
                print(f"Warning: Index {index} beyond range {len(characters)}")

        last_index = index

    # Calculate average confidence score
    score = np.mean(char_scores) if char_scores else 0

    return text, score


def process_output(conv_output_path, dict_path, weight_path):
    """
    Complete post-processing workflow

    Args:
        conv_output_path: Path to the convolution output file
        dict_path: Path to the dictionary file

    Returns:
        Recognized text and confidence score
    """

    # Load character dictionary
    characters = load_character_dict(dict_path)
    print(f"Loading {len(characters)} characers")

    # Read and dequantize convolution output
    conv_output = np.fromfile(conv_output_path, dtype=np.int8).reshape(1, 2, 640, 512)
    conv_output = dequantize(conv_output, 0.018496015879112905)
    conv_output = conv_output.permute(0, 3, 1, 2)

    # Compute recognition result probabilities from convolution output
    model_output = conv2realoutput(conv_output, weight_path)
    model_output = model_output.detach().numpy()

    print(f"Output shape of Model: {model_output.shape}")

    # Remove batch dimension
    if len(model_output.shape) == 3 and model_output.shape[0] == 1:
        model_output = model_output[0]

    # Perform CTC decoding
    text, confidence = ctc_decode(model_output, characters)

    return text, confidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model postprocess")
    parser.add_argument("--model_output_path", required=True, help="Conv2d output directory")
    parser.add_argument("--dict_path", required=True, help="file directory")
    parser.add_argument("--weight_path", required=True, help="weight file directory")

    args = parser.parse_args()

    # Perform complete post-processing
    text, confidence = process_output(args.model_output_path, args.dict_path, args.weight_path)

    print(f"\nResult: {text}")
    print(f"Confidence level: {confidence:.4f}")
