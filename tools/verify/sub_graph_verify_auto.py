import os
import shutil
import argparse
import numpy as np

def load_data(bin_file, shape, dtype):
    data = np.fromfile(bin_file, dtype=dtype)
    data = data.reshape(shape)
    return data

def cos_sim(a, b):
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    # dot
    dot_product = np.dot(a, b)

    # mod
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    cosine_similarity = dot_product / (norm_a * norm_b)

    return round(cosine_similarity, 8)

def int8_distribution_to_probability(data):
    """
    int8 to probability
    """
    counts = np.bincount(data, minlength=256)
    probabilities = counts / counts.sum()
    return probabilities

def fp32_distribution_to_probability(data, bins=256):
    """
    fp32 to probability
    data: FP32
    bins: bucket nums
    """

    min_val = np.min(data)
    max_val = np.max(data)

    # use np.histogram discretize the data into a specified number of buckets.
    counts, bin_edges = np.histogram(data, bins=bins, range=(min_val, max_val))

    # Convert frequencies into a probability distribution.
    probabilities = counts / counts.sum()

    return probabilities, bin_edges

def kl_divergence(p, q, epsilon = 1e-8):
    if p.dtype == np.int8 and q.dtype == np.int8:
        p = np.array(int8_distribution_to_probability(p), dtype=np.float64)
        q = np.array(int8_distribution_to_probability(q), dtype=np.float64)
    else:
        raise ValueError(f"Not support other type fo data P:{p.dtype} Q:{q.dtype}" )

    # normalization
    p = p / np.sum(p)
    q = q / np.sum(q)

    # avoid div the 0
    p = np.maximum(p, epsilon)
    q = np.maximum(q, epsilon)

    kl = np.sum(p * np.log(p / q))
    return kl

# Step 1: Copy the newly generated Vela model and replace the original vela.bin
def prepare_model_and_input(workdir, input_tensor):
    try:
        # Copy vela model
        source_vela_bin = os.path.join(workdir, "out_vela.bin")
        destination_vela_bin = os.path.join(os.getcwd(), "verify/vela.bin")
        shutil.copy2(source_vela_bin, destination_vela_bin)
        print(f"Successfully copied {source_vela_bin} to {destination_vela_bin}")

        # Convert the input tensor to binary
        data = np.load(f"{input_tensor}")
        data = data.transpose(0, 2, 3, 1)
        save_input_tensor_path = os.path.join(os.getcwd(), "verify/input_tensor.bin")
        data.tofile(save_input_tensor_path)
        print(f"Successfully converted {input_tensor} to {save_input_tensor_path}")
    except FileNotFoundError:
        print(f"File {source_vela_bin} not found. Please check the path.")
    except Exception as e:
        print(f"An error occurred while copying the file: {e}")

# Step 2: Build & Run
def build_and_run():
    try:
        # Execute paddle_verify_bin.sh for the first time
        tool_path = os.path.join(os.getcwd(), "tools/verify/paddle_verify_bin.sh")
        workdir = "verify"
        cmd = f"{tool_path} {workdir}"
        print(f"Execute the following command: {cmd}")
        os.system(cmd)
        print("First execution of paddle_verify_bin.sh completed")
        with open("verify/paddle_runner.txt", "r") as f:
            lines = f.readlines()
            for line in lines:
                if "output bin" in line:
                    output_bin_offset = line.split()[2][1:]
                    output_bin_size = line.split()[3][:-1]
                    break
        print(f"output bin: {output_bin_offset}, size: {output_bin_size}")
        cmd = f"{tool_path} {workdir} {output_bin_offset} {output_bin_size}"
        print(f"Execute the following command: {cmd}")
        # Execute paddle_verify_bin.sh for the second time
        os.system(cmd)
        print("Second execution of paddle_verify_bin.sh completed")
    except Exception as e:
        print(f"An error occurred while executing Step 3: {e}")

# Step 3: Calculate cosine similarity and KL divergence
def verify(gt_out, shape_str):
    running_out = "verify/output_tensor.bin"
    if shape_str == "":
        # PP_TinyPose_128x96_qat_dis_nopact_opt/pose/gt/1_conv2d_int8_output_tensor_1x32x64x48.npy
        test_shape = list(map(int, gt_out.split("/")[-1].split("_")[-1].split(".")[0].split("x")))
    else:
        test_shape = list(map(int,shape_str.split(",")))

    index_mapping = [0, 2, 3, 1]
    test_shape = [test_shape[i] for i in index_mapping]
    print(f"GroundTruth file: {gt_out}, Infer Result file{running_out} Shape:{test_shape}")

    bench_data = np.load(gt_out, allow_pickle=True)
    running_data = load_data(running_out, test_shape, np.int8)
    running_data = running_data.transpose(0, 3, 1, 2) # NHWC -> NCHW

    if running_data.dtype != bench_data.dtype:
        raise ValueError(f"Data type mismatch: {running_data.dtype} vs {bench_data.dtype}")
    elif running_data.shape != bench_data.shape:
        raise ValueError(f"Data shape mismatch: {running_data.shape} vs {bench_data.shape}")

    # np.set_printoptions(threshold=sys.maxsize)
    print(f"bdata: {bench_data.shape}, rdata: {running_data.shape}")
    print(f"run out:{(bench_data - running_data).flatten()}")
    print(f"mean:{np.mean(bench_data - running_data)}, std:{np.std(bench_data - running_data)} max:{np.max(bench_data - running_data)}, min:{np.min(bench_data - running_data)}")
    print(f"cos sim {cos_sim(bench_data.flatten(), running_data.flatten())}")
    print(f"KL Divergence {kl_divergence(bench_data.flatten(), running_data.flatten())}")

def init_args():
    parser = argparse.ArgumentParser(description='Execute the model verification process')
    parser.add_argument('--workdir', default='tinypose_output', help='Path to the working directory')
    # Step 2 parameters
    parser.add_argument('--input_tensor',
                        default='model_zoo/PP_TinyPose_128x96_qat_dis_nopact_opt/pose/gt/1_conv2d_int8_input_tensor_1x16x64x48.npy',
                        help='Path to the source input tensor .npy file')
    parser.add_argument('--gt',
                        default='model_zoo/PP_TinyPose_128x96_qat_dis_nopact_opt/pose/gt/1_conv2d_int8_output_tensor_1x32x64x48.npy',
                        help='Path to the source output tensor .npy file')
    parser.add_argument('--shape_str', default='', help='Shape string')
    return parser.parse_args()

def main():
    args = init_args()
    # Copy the newly generated Vela model and replace the original vela.bin
    prepare_model_and_input(args.workdir, args.input_tensor)
    print("\n")
    build_and_run()
    print("\n")
    verify(args.gt, args.shape_str)

if __name__ == "__main__":
    main()
