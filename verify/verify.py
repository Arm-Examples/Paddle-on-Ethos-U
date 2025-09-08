import numpy as np
import sys

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

if __name__ == "__main__":
    if len(sys.argv) < 4:
        raise ValueError("Usage: python verify.py benchmark.npy running_out.bin test_shape")

    groundtruth_out = sys.argv[1]
    running_out = sys.argv[2]
    test_shape = list(map(int, sys.argv[3].split(",")))
    print(f"GroundTruth file: {groundtruth_out}, Infer Result file{running_out} Shape:{test_shape}")

    bench_data = np.load(groundtruth_out, allow_pickle=True)
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
