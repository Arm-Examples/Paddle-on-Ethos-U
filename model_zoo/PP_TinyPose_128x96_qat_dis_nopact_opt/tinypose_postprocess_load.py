import argparse
import numpy as np
import cv2
import math
import os


SCORE_THRESHOLD = 0.001
SEARCH_SCOPE = 4

def get_keypoints(score_map, num_joints, score_threshold):
    """Get Keypoint"""
    result = {"keypoints": [], "scores": [], "num_joints": num_joints, "kpt_count": 0}

    # Find the rough center point
    x, y = 0, 0
    max_score = -1.0
    for i in range(32):
        for j in range(24):
            if score_map[i, j] > max_score:
                x, y = i, j
                max_score = score_map[i, j]

    # Return empty result if no keypoint is found
    if max_score < score_threshold:
        result["num_joints"] = -1
        return result

    # Get potential points
    search_scope = min(SEARCH_SCOPE, x, y, 31 - x, 23 - y)
    for i in range(x - search_scope, x + search_scope + 1):
        for j in range(y - search_scope, y + search_scope + 1):
            result["keypoints"].append([float(i), float(j)])
            result["scores"].append(score_map[i, j])
            result["kpt_count"] += 1

    return result


def find_center(result, scale_x, scale_y):
    """Find the center point"""
    sum_scores = 0.0
    x, y = 0.0, 0.0

    for i in range(result["kpt_count"]):
        pow_scores = math.pow(result["scores"][i], 3)
        x += pow_scores * result["keypoints"][i][0]
        y += pow_scores * result["keypoints"][i][1]
        sum_scores += pow_scores

    return [
        (x / sum_scores + 0.5) * scale_x - 0.5,
        (y / sum_scores + 0.5) * scale_y - 0.5,
    ]


def postprocess(output_data, score_threshold, output_image):
    """Post-processing function"""
    results = []
    target_detected = True

    # Scaling factors
    scale_x = output_image.shape[0] / 32.0
    scale_y = output_image.shape[1] / 24.0

    # Keypoint connection definitions
    link_kpt = [
        [0, 1],
        [1, 3],
        [3, 5],
        [5, 7],
        [7, 9],
        [5, 11],
        [11, 13],
        [13, 15],
        [0, 2],
        [2, 4],
        [4, 6],
        [6, 8],
        [8, 10],
        [6, 12],
        [12, 14],
        [14, 16],
        [11, 12],
    ]

    # Get post-processing results
    output_size = output_data.shape[1] * output_data.shape[2] * output_data.shape[3]
    for i in range(0, output_size, 768):
        joint_idx = i // 768
        score_map = output_data[0, joint_idx].reshape(32, 24)
        result = get_keypoints(score_map, joint_idx, score_threshold)
        results.append(result)
        if result["num_joints"] < 0:
            target_detected = False
            break

    # Display results
    if target_detected:
        kpts = []
        print(f"results: {len(results)}")

        for i in range(len(results)):
            center = find_center(results[i], scale_x, scale_y)
            kpts.append((int(center[1]), int(center[0])))
            print(f"[{results[i]['num_joints']}] - {center[0]}, {center[1]}")

        # Draw keypoints and connections
        for p in kpts:
            cv2.circle(output_image, p, 2, (0, 0, 255), -1)

        for idx in link_kpt:
            if idx[0] < len(kpts) and idx[1] < len(kpts):
                cv2.line(output_image, kpts[idx[0]], kpts[idx[1]], (255, 0, 0), 1)

    return results


def main():
    parser = argparse.ArgumentParser(description="Pose Detection Post-processing")
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument("--confidence", type=float, default=0.01, help="Score confidence threshold")
    parser.add_argument(
        "--output", required=True, help="Model output binary file (.bin)"
    )
    parser.add_argument(
        "--save", default="./model_zoo/PP_TinyPose_128x96_qat_dis_nopact_opt/output.jpg", help="Save path for annotated image"
    )
    args = parser.parse_args()

    # Read input image
    input_image = cv2.imread(args.image)
    if input_image is None:
        print(f"Failed to read image: {args.image}")
        return

    output_image = input_image.copy()

    # Read model output data (binary file)
    try:
        output_data = np.fromfile(args.output, dtype=np.int8)
        output_data = output_data.reshape(1, 32, 24, 17)
        output_data = np.transpose(output_data, (0, 3, 1, 2))
        output_data = output_data * 0.007520623
    except Exception as e:
        print(f"Failed to load model output: {e}")
        return

    print(f"Processing image: {args.image}")
    results = postprocess(output_data, args.confidence, output_image)

    # Save result image
    cv2.imwrite(args.save, output_image)
    print(f"Result saved to: {args.save}")


if __name__ == "__main__":
    main()
