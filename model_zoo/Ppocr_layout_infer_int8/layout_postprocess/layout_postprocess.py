import numpy as np
import cv2
import argparse
import torch
import os 


def nms_per_class(boxes, scores, iou_threshold, score_threshold, max_output_boxes):
    """
    Implementation of Non-Maximum Suppression (NMS) for a single class.

    Args:
    - boxes: Array of shape [N, 4], where each row is [x1, y1, x2, y2]
    - scores: Array of shape [N], representing the confidence score of each bounding box
    - iou_threshold: IoU threshold
    - score_threshold: Score threshold
    - max_output_boxes: Maximum number of output bounding boxes

    Returns:
    - Indices of the retained bounding boxes
    """
    valid_mask = scores > score_threshold
    boxes = boxes[valid_mask]
    scores = scores[valid_mask]

    if len(boxes) == 0:
        return np.array([], dtype=np.int32)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)

    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0 and len(keep) < max_output_boxes:
        i = order[0]
        keep.append(i)
        if order.size == 1 or len(keep) == max_output_boxes:
            break

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= iou_threshold)[0]
        order = order[inds + 1]

    original_indices = np.where(valid_mask)[0][keep]
    return original_indices


def multi_class_nms(
    boxes, scores, iou_threshold, score_threshold, max_output_boxes_per_class
):
    """
    Multi-class NMS

    Args:
    - boxes: Array of shape [batch_size, num_boxes, 4]
    - scores: Array of shape [batch_size, num_classes, num_boxes]
    - iou_threshold: IoU threshold
    - score_threshold: Score threshold
    - max_output_boxes_per_class: Maximum number of output bounding boxes per class

    Returns:
    - Array of shape [N, 3], where each row is [batch_index, class_index, box_index]
    """
    batch_size = boxes.shape[0]
    num_classes = scores.shape[1]

    selected_indices = []

    for batch_idx in range(batch_size):
        batch_boxes = boxes[batch_idx]  # [num_boxes, 4]

        for class_idx in range(num_classes):
            class_scores = scores[batch_idx, class_idx]  # [num_boxes]

            keep_indices = nms_per_class(
                batch_boxes,
                class_scores,
                iou_threshold,
                score_threshold,
                max_output_boxes_per_class,
            )

            for box_idx in keep_indices:
                selected_indices.append([batch_idx, class_idx, box_idx])

    if len(selected_indices) == 0:
        return np.array([], dtype=np.int32).reshape(0, 3)

    return np.array(selected_indices, dtype=np.int32)


def softmax(x, dim=1):
    x = x - np.max(x, axis=dim, keepdims=True)
    x = np.exp(x) / np.sum(np.exp(x), axis=dim, keepdims=True)
    return x


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def load_labels(label_file):
    with open(label_file, "r") as f:
        return [line.strip() for line in f.readlines()]


def process_outputs(output_paths, bias_paths):
    output0 = np.fromfile(output_paths[0], dtype=np.float32)
    output1 = np.fromfile(output_paths[1], dtype=np.float32)
    output2 = np.fromfile(output_paths[2], dtype=np.float32)
    output3 = np.fromfile(output_paths[3], dtype=np.float32)

    bias0 = np.load(bias_paths[0]).reshape(1, 37, 1, 1)
    bias1 = np.load(bias_paths[1]).reshape(1, 37, 1, 1)
    bias2 = np.load(bias_paths[2]).reshape(1, 37, 1, 1)
    bias3 = np.load(bias_paths[3]).reshape(1, 37, 1, 1)

    output0 = output0.reshape(1, 37, 100, 76)
    output1 = output1.reshape(1, 37, 50, 38)
    output2 = output2.reshape(1, 37, 25, 19)
    output3 = output3.reshape(1, 37, 13, 10)

    output0 = output0 + bias0
    output1 = output1 + bias1
    output2 = output2 + bias2
    output3 = output3 + bias3

    output0_1 = output0[:, :5, :, :]
    output0_0 = output0[:, 5:, :, :]
    output1_1 = output1[:, :5, :, :]
    output1_0 = output1[:, 5:, :, :]
    output2_1 = output2[:, :5, :, :]
    output2_0 = output2[:, 5:, :, :]
    output3_1 = output3[:, :5, :, :]
    output3_0 = output3[:, 5:, :, :]

    output0_0 = np.transpose(output0_0, (0, 2, 3, 1)).reshape(-1, 8)
    output1_0 = np.transpose(output1_0, (0, 2, 3, 1)).reshape(-1, 8)
    output2_0 = np.transpose(output2_0, (0, 2, 3, 1)).reshape(-1, 8)
    output3_0 = np.transpose(output3_0, (0, 2, 3, 1)).reshape(-1, 8)

    output0_0 = softmax(output0_0) @ np.array(
        [0, 1, 2, 3, 4, 5, 6, 7], dtype=np.float32
    )
    output1_0 = softmax(output1_0) @ np.array(
        [0, 1, 2, 3, 4, 5, 6, 7], dtype=np.float32
    )
    output2_0 = softmax(output2_0) @ np.array(
        [0, 1, 2, 3, 4, 5, 6, 7], dtype=np.float32
    )
    output3_0 = softmax(output3_0) @ np.array(
        [0, 1, 2, 3, 4, 5, 6, 7], dtype=np.float32
    )

    output0_0 = output0_0.reshape(1, 7600, 4)
    output1_0 = output1_0.reshape(1, 1900, 4)
    output2_0 = output2_0.reshape(1, 475, 4)
    output3_0 = output3_0.reshape(1, 130, 4)

    output0_1 = output0_1.reshape(1, 5, 7600)
    output1_1 = output1_1.reshape(1, 5, 1900)
    output2_1 = output2_1.reshape(1, 5, 475)
    output3_1 = output3_1.reshape(1, 5, 130)

    output0_1 = sigmoid(output0_1)
    output1_1 = sigmoid(output1_1)
    output2_1 = sigmoid(output2_1)
    output3_1 = sigmoid(output3_1)

    output1 = np.concatenate((output0_1, output1_1, output2_1, output3_1), axis=2)

    output0 = np.concatenate((output0_0, output1_0, output2_0, output3_0), axis=1)
    output0_0 = output0[:, :, 2:] + np.load("./model_zoo/Ppocr_layout_infer_int8/layout_postprocess/c1.npy").reshape(1, 10105, 2)
    output0_1 = -output0[:, :, :2] + np.load("./model_zoo/Ppocr_layout_infer_int8/layout_postprocess/c2.npy").reshape(1, 10105, 2)
    output0 = np.concatenate((output0_1, output0_0), axis=2) * np.load(
        "./model_zoo/Ppocr_layout_infer_int8/layout_postprocess/c3.npy"
    ).reshape(1, 10105, 1)

    return output0, output1


def perform_nms(
    output0, output1, iou_threshold=0.6, score_threshold=0.025, max_boxes=100
):
    selected = multi_class_nms(
        output0,
        output1,
        iou_threshold=iou_threshold,
        score_threshold=score_threshold,
        max_output_boxes_per_class=1000,
    )

    gather0 = selected[:, 2]
    gather1 = selected[:, 1]
    tmp1 = gather1 * 10105 + gather0
    tmp1 = tmp1.reshape(-1)
    tmp2 = output1.reshape(-1)
    tmp3 = tmp2[tmp1]

    num = min(tmp3.shape[0], max_boxes)
    indices = torch.topk(
        torch.tensor(tmp3), num, sorted=True, largest=True
    ).indices.numpy()

    tmp4 = tmp3[indices].reshape(1, -1, 1)
    tmp5 = gather1.reshape(-1)[indices].reshape(1, -1, 1)
    tmp6 = gather0.reshape(-1)[indices]
    tmp7 = output0.reshape(1, -1, 4)[:, tmp6, :]

    output = np.concatenate((tmp7, tmp4, tmp5), axis=2)

    return output


def visualize_detection(image_path, output, class_names, thresh=0.5):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Can not read Image: {image_path}")
        return

    oriw = image.shape[1]
    orih = image.shape[0]

    detected_count = 0

    for i in range(output.shape[1]):
        box = output[0, i, :]

        class_id = int(round(box[5]))
        score = box[4]
        print(f"Detection Class ID: {class_id}, Confidence Level: {score}")
        if score <= thresh:
            continue

        detected_count += 1

        xmin = int(box[0])
        ymin = int(box[1])
        xmax = int(box[2])
        ymax = int(box[3])
        w = xmax - xmin
        h = ymax - ymin

        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(oriw, xmax)
        ymax = min(orih, ymax)
        w = xmax - xmin
        h = ymax - ymin

        if w <= 0 or h <= 0 or score > 1:
            continue

        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2, cv2.LINE_AA)

        class_name = class_names[class_id] if class_id < len(class_names) else "Unkown"
        text = f"{class_name}: {score:.3f}"

        font_face = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1

        text_size = cv2.getTextSize(text, font_face, font_scale, thickness)[0]

        scale_factor = max(float(w), 200) * 0.35 * font_scale / text_size[0]
        new_font_scale = min(scale_factor, 1.0)
        text_size = cv2.getTextSize(text, font_face, new_font_scale, thickness)[0]

        origin_x = xmin + 5
        origin_y = ymin + text_size[1] + 5

        cv2.rectangle(
            image,
            (origin_x - 2, origin_y - text_size[1] - 2),
            (origin_x + text_size[0] + 2, origin_y + 2),
            (0, 255, 0),
            -1,
        )

        cv2.putText(
            image,
            text,
            (origin_x, origin_y),
            font_face,
            new_font_scale,
            (0, 0, 0),
            thickness,
            cv2.LINE_AA,
        )

        print(
            f"Detection Result - {class_name}, Confidence Level: {score:.3f}, "
            f"Position: x={xmin}, y={ymin}, Width={w}, Height={h}"
        )

    print(f"Detect {detected_count} targets")

    img_name = os.path.splitext(os.path.basename(image_path))[0]
    result_name = f"{img_name}_object_detection_result.jpg"
    result_name = "model_zoo/Ppocr_layout_infer_int8/" + result_name
    cv2.imwrite(result_name, image)
    print(f"Result Image saved into : {result_name}")

    return image


def main():
    parser = argparse.ArgumentParser(description="Target detection")
    parser.add_argument("--image", type=str, required=True, help="Input Path")
    parser.add_argument(
        "--labels",
        type=str,
        default="./model_zoo/Ppocr_layout_infer_int8/layout_postprocess/layout_labels.txt",
        help="label path",
    )
    parser.add_argument("--confidence", type=float, default=0.42, help="Confidence threshold")
    parser.add_argument("--iou_threshold", type=float, default=0.6, help="IoU threshold")
    parser.add_argument(
        "--output_dir", type=str, default="./output", help="Output path"
    )
    args = parser.parse_args()

    output_paths = [
        os.path.join(args.output_dir, "output_0.bin"),
        os.path.join(args.output_dir, "output_1.bin"),
        os.path.join(args.output_dir, "output_2.bin"),
        os.path.join(args.output_dir, "output_3.bin"),
    ]

    bias_paths = [
        "./model_zoo/Ppocr_layout_infer_int8/layout_postprocess/bias_1x37x100x76.npy",
        "./model_zoo/Ppocr_layout_infer_int8/layout_postprocess/bias_1x37x50x38.npy",
        "./model_zoo/Ppocr_layout_infer_int8/layout_postprocess/bias_1x37x25x19.npy",
        "./model_zoo/Ppocr_layout_infer_int8/layout_postprocess/bias_1x37x13x10.npy",
    ]

    output0, output1 = process_outputs(output_paths, bias_paths)

    final_output = perform_nms(
        output0,
        output1,
        iou_threshold=args.iou_threshold,
        score_threshold=args.confidence,
    )

    class_names = load_labels(args.labels)

    # Visualize the detection results
    visualize_detection(args.image, final_output, class_names, args.confidence)


if __name__ == "__main__":
    main()
