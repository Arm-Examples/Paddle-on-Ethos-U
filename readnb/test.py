import argparse
import sys

from inner_test.shape_test import shape_compare
parser = argparse.ArgumentParser()
parser.add_argument(
    "--bg_file", type=str, required=True, help="Groundtruth file path")
parser.add_argument(
    "--model_file", default="", type=str, help="Model src file path")
parser.add_argument(
    "--test", default="shape", type=str, help="test forms")

if __name__ == "__main__":
    args = parser.parse_args()
    bg_file = args.bg_file
    model = args.model_file
    test_mode = args.test

    if test_mode ==  "shape":
        shape_compare.infer(bg_file, model)
    else:
        raise RuntimeError(f"Not support test {test_mode}")


