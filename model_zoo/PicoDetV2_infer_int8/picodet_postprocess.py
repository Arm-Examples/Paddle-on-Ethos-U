#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
import argparse
from typing import List, Tuple, Dict


def load_labels(label_file: str) -> List[str]:
    """Load label file"""

    with open(label_file, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines()]


def read_binary_output(output_index: int, output_dir: str = ".") -> np.ndarray:
    """
    Read binary output file with fixed shape

    Args:
        output_index: Output index (0-7)
        output_dir: Directory containing output files

    Returns:
        Reshaped data array
    """
    # Define fixed output shapes for PicoDet
    # 0-3: Classification output - feature map shape is (1, num_points, 80)
    # 4-7: Regression output - feature map shape is (1, num_points, 32)
    feature_sizes = [(52, 52), (26, 26), (13, 13), (7, 7)]

    if output_index < 4:
        # Shape: (1, h*w, 80)
        h, w = feature_sizes[output_index]
        shape = (1, h * w, 80)
    else:  # Regression output
        # Sape: (1, h*w, 32)
        h, w = feature_sizes[output_index - 4]
        shape = (1, h * w, 32)

    # Read binary data
    bin_file = os.path.join(output_dir, f"output_{output_index}.bin")
    data = np.fromfile(bin_file, dtype=np.float32)

    # Ensure data size matches the expected shape
    expected_size = np.prod(shape)
    if len(data) != expected_size:
        raise ValueError(f"Data size mismatch: expected {expected_size}, got {len(data)}")

    return data.reshape(shape)


class BoxInfo:
    """Detection box information class"""

    def __init__(self, x1, y1, x2, y2, score, label):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.score = score
        self.label = label


def softmax(x):
    """Apply softmax normalization to data"""
    x_max = np.max(x, axis=-1, keepdims=True)
    e_x = np.exp(x - x_max)
    return e_x / np.sum(e_x, axis=-1, keepdims=True)


def decode_infer(
    cls_pred,
    dis_pred,
    stride,
    threshold,
    feature_h,
    feature_w,
    img_w,
    img_h,
    num_class,
    reg_max,
):
    """
    Decode inference results

    Args:
        cls_pred: Classification predictions (n, num_class)
        dis_pred: Distance predictions (n, 4*(reg_max+1))
        stride: Stride
        threshold: Confidence threshold
        feature_h: Feature map height
        feature_w: Feature map width
        img_w: Image width
        img_h: Image height
        num_class: Number of classes
        reg_max: Maximum regression value

    Returns:
        List of detection boxes grouped by class
    """
    # Create results grouped by class
    results = [[] for _ in range(num_class)]

    # Iterate over each anchor point
    for idx in range(feature_h * feature_w):
        # Calculate grid coordinates
        row = idx // feature_w
        col = idx % feature_w

        # Get classification predictions
        scores = cls_pred[idx]

        # Find the highest score and its corresponding class
        score = np.max(scores)
        cur_label = np.argmax(scores)

        # Output some debug information for every 500th sample
        if idx % 500 == 0:
            print(
                f"Sample idx={idx}, row={row}, col={col}, score={score:.4f}, class={cur_label}"
            )

        if score > threshold:
            bbox_pred = dis_pred[idx].reshape(4, reg_max + 1)

            # Calculate center point coordinates
            ct_x = (col + 0.5) * stride
            ct_y = (row + 0.5) * stride

            # Decode distances in four directions
            dis_pred_decoded = np.zeros(4)
            for i in range(4):
                # Current direction prediction
                dis_pred_item = bbox_pred[i]

                dis_after_sm = softmax(dis_pred_item)

                # Calculate weighted sum
                dis = 0
                for j in range(reg_max + 1):
                    dis += j * dis_after_sm[j]

                # Multiply by stride to get actual distance
                dis_pred_decoded[i] = dis * stride

            # Calculate detection box coordinates
            xmin = max(ct_x - dis_pred_decoded[0], 0.0)
            ymin = max(ct_y - dis_pred_decoded[1], 0.0)
            xmax = min(ct_x + dis_pred_decoded[2], img_w)
            ymax = min(ct_y + dis_pred_decoded[3], img_h)

            # Check box validity
            if xmin >= xmax or ymin >= ymax:
                continue

            # Create detection box object and add to the corresponding class result list
            box = BoxInfo(xmin, ymin, xmax, ymax, score, cur_label)
            results[cur_label].append(box)

            if len(results[cur_label]) % 20 == 0:
                print(f"Class {cur_label} has {len(results[cur_label])} boxes")

    return results


def calculate_iou(box1, box2):
    """Calculate IoU (Intersection over Union) between two boxes"""
    x1 = max(box1.x1, box2.x1)
    y1 = max(box1.y1, box2.y1)
    x2 = min(box1.x2, box2.x2)
    y2 = min(box1.y2, box2.y2)

    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    inter_area = w * h

    area1 = (box1.x2 - box1.x1) * (box1.y2 - box1.y1)
    area2 = (box2.x2 - box2.x1) * (box2.y2 - box2.y1)

    return inter_area / (area1 + area2 - inter_area)


def nms(input_boxes, nms_threshold):
    """Non-Maximum Suppression (NMS)"""
    if not input_boxes:
        return []

    # Sort by confidence score
    input_boxes.sort(key=lambda x: x.score, reverse=True)

    box_num = len(input_boxes)
    merged = [0] * box_num

    results = []
    for i in range(box_num):
        if merged[i]:
            continue

        results.append(input_boxes[i])

        for j in range(i + 1, box_num):
            if merged[j]:
                continue

            # Perform NMS only between same class objects
            if input_boxes[i].label != input_boxes[j].label:
                continue

            iou = calculate_iou(input_boxes[i], input_boxes[j])
            if iou > nms_threshold:
                merged[j] = 1

    return results


def draw_detection_results(image, results, class_names, threshold):
    """Draw detection results on the image"""
    for box in results:
        if box.score > threshold:
            xmin = int(box.x1)
            ymin = int(box.y1)
            xmax = int(box.x2)
            ymax = int(box.y2)

            w = xmax - xmin
            h = ymax - ymin

            # Ensure the box is within image bounds
            if xmin < 0 or ymin < 0 or xmax > image.shape[1] or ymax > image.shape[0]:
                xmin = max(0, xmin)
                ymin = max(0, ymin)
                w = min(image.shape[1] - xmin, w)
                h = min(image.shape[0] - ymin, h)

            class_name = (
                class_names[box.label] if box.label < len(class_names) else "Unknown"
            )

            if w > 0 and h > 0 and box.score <= 1:
                # Draw rectangle
                cv2.rectangle(image, (xmin, ymin), (xmin + w, ymin + h), (0, 0, 255), 2)

                # Draw class and confidence score
                str_prob = f"{box.score:.3f}"
                text = f"{class_name}: {str_prob}"
                font_face = cv2.FONT_HERSHEY_COMPLEX_SMALL
                font_scale = 1.0
                thickness = 2
                text_size = cv2.getTextSize(text, font_face, font_scale, thickness)[0]

                new_font_scale = (
                    w * 0.35 * font_scale / text_size[0]
                    if text_size[0] > 0
                    else font_scale
                )
                text_size = cv2.getTextSize(text, font_face, new_font_scale, thickness)[
                    0
                ]

                origin_x = xmin + 10
                origin_y = ymin + text_size[1] + 10

                cv2.putText(
                    image,
                    text,
                    (origin_x, origin_y),
                    font_face,
                    new_font_scale,
                    (0, 255, 255),
                    thickness,
                    cv2.LINE_AA,
                )

                print(
                    f"detection, image size: {image.shape[1]}, {image.shape[0]}, "
                    f"detect object: {class_name}, score: {box.score:.4f}, "
                    f"location: x={xmin}, y={ymin}, width={w}, height={h}"
                )


def process_picodet_output(
    output_dir,
    image_path,
    result_path,
    label_file,
    threshold=0.5,
    nms_threshold=0.3,
    input_size=(416, 416),
):
    """
    Process the output of the PicoDet model

    Args:
        output_dir: Directory containing model output files
        image_path: Path to the input image
        label_file: Path to the label file
        threshold: Confidence threshold
        nms_threshold: NMS threshold
        input_size: Model input size
    """
    # Load labels
    labels = load_labels(label_file)
    print(f"Loaded {len(labels)} labels")

    # Read the input image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Load image failed: {image_path}")

    ori_height, ori_width = image.shape[:2]
    input_width, input_height = input_size

    # Feature map sizes and strides
    feature_sizes = [(52, 52), (26, 26), (13, 13), (7, 7)]
    strides = [8, 16, 32, 64]

    # Fixed index mapping arrays
    cls_indices = [0, 1, 2, 3]  # Classification output indices
    reg_indices = [4, 5, 6, 7]  # Regression output indices

    reg_max = 7  # P2B model typically uses 7
    num_class = 80

    # Store detection boxes grouped by class
    all_boxes_by_class = [[] for _ in range(num_class)]

    # Process four feature layers
    for i in range(4):
        try:
            # Read classification and regression outputs
            cls_data = read_binary_output(cls_indices[i], output_dir)
            dis_data = read_binary_output(reg_indices[i], output_dir)

            # Get feature map dimensions
            feature_h, feature_w = feature_sizes[i]

            # Remove batch dimension and ensure correct shape
            cls_data = cls_data.reshape(-1, 80)
            dis_data = dis_data.reshape(-1, 4 * (reg_max + 1))

            print(f"Feature layer {i} (stride={strides[i]}):")
            print(f"  cls shape: {cls_data.shape}")
            print(f"  dis shape: {dis_data.shape}")

            # Decode the current feature layer
            layer_results = decode_infer(
                cls_data,
                dis_data,
                strides[i],
                threshold,
                feature_h,
                feature_w,
                input_width,
                input_height,
                num_class,
                reg_max,
            )

            # Merge into the overall grouped detection boxes by class
            for cls in range(num_class):
                all_boxes_by_class[cls].extend(layer_results[cls])

        except Exception as e:
            print(f"Error processing feature layer {i}: {e}")

    # Perform NMS for each class
    total_before_nms = sum(len(boxes) for boxes in all_boxes_by_class)

    result_list = []
    for cls in range(num_class):
        if all_boxes_by_class[cls]:
            filtered_boxes = nms(all_boxes_by_class[cls], nms_threshold)

            # Map coordinates to the original image size
            for box in filtered_boxes:
                box.x1 = box.x1 / input_width * ori_width
                box.x2 = box.x2 / input_width * ori_width
                box.y1 = box.y1 / input_height * ori_height
                box.y2 = box.y2 / input_height * ori_height
                result_list.append(box)

    total_after_nms = len(result_list)
    print(f"Total boxes before NMS: {total_before_nms}")
    print(f"Total boxes after NMS: {total_after_nms}")

    # Draw detection results
    draw_detection_results(image, result_list, labels, threshold)

    result_name = (
        os.path.splitext(os.path.basename(image_path))[0] + "_detection_result.jpg"
    )
    result_name = result_path + result_name

    cv2.imwrite(result_name, image)
    print(f"Results saved to: {result_name}")

    return image, result_list


def main():
    parser = argparse.ArgumentParser(description="PicoDetection Post-Processing Tool")
    parser.add_argument(
        "--output_dir", type=str, default=".", help="Directory containing model output files"
    )
    parser.add_argument("--image", type=str, required=True, help="Path to the input image")
    parser.add_argument("--result", type=str, help="Result output path")
    parser.add_argument("--label_file", type=str, required=True, help="Path to the label file")
    parser.add_argument("--confidence", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--nms_threshold", type=float, default=0.3, help="NMS threshold")
    parser.add_argument("--width", type=int, default=416, help="Model input width")
    parser.add_argument("--height", type=int, default=416, help="Model input height")

    args = parser.parse_args()

    process_picodet_output(
        args.output_dir,
        args.image,
        args.result,
        args.label_file,
        args.confidence,
        args.nms_threshold,
        (args.width, args.height),
    )


if __name__ == "__main__":
    main()
