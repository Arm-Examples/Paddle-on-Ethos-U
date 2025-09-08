import os
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.special import expit


def decode_boxes(
    raw_boxes, anchors, x_scale=256.0, y_scale=256.0, w_scale=256.0, h_scale=256.0
):
    boxes = np.zeros_like(raw_boxes)

    x_center = raw_boxes[..., 0] / x_scale * anchors[:, 2] + anchors[:, 0]
    y_center = raw_boxes[..., 1] / y_scale * anchors[:, 3] + anchors[:, 1]

    w = raw_boxes[..., 2] / w_scale * anchors[:, 2]
    h = raw_boxes[..., 3] / h_scale * anchors[:, 3]

    boxes[..., 0] = y_center - h / 2.0  # ymin
    boxes[..., 1] = x_center - w / 2.0  # xmin
    boxes[..., 2] = y_center + h / 2.0  # ymax
    boxes[..., 3] = x_center + w / 2.0  # xmax

    for k in range(6):
        offset = 4 + k * 2
        keypoint_x = raw_boxes[..., offset] / x_scale * anchors[:, 2] + anchors[:, 0]
        keypoint_y = (
            raw_boxes[..., offset + 1] / y_scale * anchors[:, 3] + anchors[:, 1]
        )
        boxes[..., offset] = keypoint_x
        boxes[..., offset + 1] = keypoint_y

    return boxes


def overlap_similarity(box, other_boxes):
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    other_boxes_area = (other_boxes[:, 2] - other_boxes[:, 0]) * (
        other_boxes[:, 3] - other_boxes[:, 1]
    )

    max_xy = np.minimum(
        np.tile(box[2:4], [other_boxes.shape[0], 1]),
        other_boxes[:, 2:4],
    )
    min_xy = np.maximum(
        np.tile(box[0:2], [other_boxes.shape[0], 1]),
        other_boxes[:, 0:2],
    )

    inter = np.clip(max_xy - min_xy, 0, None)
    inter_area = inter[:, 0] * inter[:, 1]

    union_area = box_area + other_boxes_area - inter_area

    iou = inter_area / union_area

    return iou


def weighted_non_max_suppression(detections, min_suppression_threshold=0.3):
    if detections.shape[0] == 0:
        return np.zeros([0, 17], dtype=np.float32)

    output_detections = []

    scores = detections[:, 16]
    remaining = np.argsort(scores)[::-1]
    remaining_list = remaining.tolist()

    while len(remaining_list) > 0:
        idx = remaining_list[0]
        detection = detections[idx]

        first_box = detection[:4]
        other_idxs = np.array(remaining_list, dtype=np.int64)
        other_boxes = detections[other_idxs][:, :4]
        ious = overlap_similarity(first_box, other_boxes)

        mask = ious > min_suppression_threshold

        overlapping_idxs = [remaining_list[i] for i in range(len(mask)) if mask[i]]

        remaining_list = [remaining_list[i] for i in range(len(mask)) if not mask[i]]

        if len(overlapping_idxs) > 1:
            overlapping_tensor = np.array(overlapping_idxs, dtype=np.int64)
            overlapping_detections = detections[overlapping_tensor]

            coordinates = overlapping_detections[:, :16]
            scores = overlapping_detections[:, 16:17]
            total_score = np.sum(scores)
            weighted = np.sum(coordinates * scores, axis=0) / total_score

            weighted_detection = np.concatenate(
                [
                    weighted,
                    np.array([total_score / len(overlapping_idxs)], dtype=np.float32),
                ]
            )
        else:
            weighted_detection = detection

        output_detections.append(weighted_detection)

    if output_detections:
        return np.stack(output_detections)
    else:
        return np.zeros([0, 17], dtype=np.float32)


def process_detections(
    raw_boxes, raw_scores, anchors, min_score_thresh=0.65, min_suppression_threshold=0.3
):
    boxes = decode_boxes(raw_boxes, anchors)

    scores = np.clip(raw_scores, -100.0, 100.0)
    scores = expit(scores).squeeze(axis=-1)

    mask = scores[0] >= min_score_thresh
    indices = np.nonzero(mask)[0]

    if indices.shape[0] == 0:
        return np.zeros([0, 17], dtype=np.float32)

    filtered_boxes = boxes[0, indices]
    filtered_scores = scores[0, indices].reshape(-1, 1)

    detections = np.concatenate([filtered_boxes, filtered_scores], axis=-1)

    return weighted_non_max_suppression(detections, min_suppression_threshold)


def visualize_detections(img, detections, output_path=None, with_keypoints=True):
    import os
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import numpy as np

    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.axis("off")

    if (
        isinstance(detections, np.ndarray)
        and detections.ndim == 1
        and detections.shape[0] > 0
    ):
        detections = np.expand_dims(detections, axis=0)

    print(f"Found {detections.shape[0]} faces")

    for i in range(detections.shape[0]):
        ymin = detections[i, 0] * img.shape[0]
        xmin = detections[i, 1] * img.shape[1]
        ymax = detections[i, 2] * img.shape[0]
        xmax = detections[i, 3] * img.shape[1]

        rect = patches.Rectangle(
            (xmin, ymin),
            xmax - xmin,
            ymax - ymin,
            linewidth=2,
            edgecolor="r",
            facecolor="none",
            alpha=0.7,
        )
        plt.gca().add_patch(rect)

        plt.text(
            xmin,
            ymin - 5,
            f"Score: {detections[i, 16]:.2f}",
            color="white",
            fontsize=12,
            bbox=dict(facecolor="red", alpha=0.5),
        )

        if with_keypoints:
            for k in range(6):
                kp_x = detections[i, 4 + k * 2] * img.shape[1]
                kp_y = detections[i, 4 + k * 2 + 1] * img.shape[0]
                plt.plot(
                    kp_x,
                    kp_y,
                    "o",
                    markersize=4,
                    markerfacecolor="cyan",
                    markeredgecolor="blue",
                )

    if output_path:
        dirname = os.path.dirname(output_path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        plt.savefig(output_path)
        print(f"Result saved to : {output_path}")
    else:
        plt.show()


def debug_model_output(raw_boxes, raw_scores):
    print("\n===== Model output  =====")
    print(f"raw_boxes shape: {raw_boxes.shape}")
    print(f"raw_scores shape: {raw_scores.shape}")

    print(f"raw_boxes range: {np.min(raw_boxes)} to {np.max(raw_boxes)}")
    print(f"raw_scores range: {np.min(raw_scores)} to {np.max(raw_scores)}")

    scores = expit(raw_scores).squeeze()
    print(f"sigmoid score range: {np.min(scores)} to {np.max(scores)}")

    thresholds = [0.1, 0.3, 0.5, 0.65, 0.8, 0.9]
    for thresh in thresholds:
        count = np.sum(scores >= thresh)
        print(f"Score >= {thresh} Number: {count}")

    print("=========================\n")


def process_face_detection(
    boxes_file,
    scores_file,
    image_path,
    anchors_path,
    output_path=None,
    min_score_thresh=0.8,
):
    try:
        raw_boxes = np.fromfile(boxes_file, dtype=np.float32)
        raw_scores = np.fromfile(scores_file, dtype=np.float32)

        num_anchors = raw_boxes.size // 16
        raw_boxes = raw_boxes.reshape(1, num_anchors, 16)
        raw_scores = raw_scores.reshape(1, num_anchors, 1)

        print(f"Output - boxes: {raw_boxes.shape}, scores: {raw_scores.shape}")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    try:
        anchors = np.load(anchors_path)
        print(f"Load success: {anchors_path}")
    except Exception as e:
        print(f"Load failed: {e}")
        return

    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"Can not read image: {image_path}")
            return

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        print(f"Success load image: {image_path}, shape: {img.shape}")
    except Exception as e:
        print(f"read image failed: {e}")
        return

    img_resized = cv2.resize(img, (256, 256))

    debug_model_output(raw_boxes, raw_scores)

    try:
        detections = process_detections(
            raw_boxes,
            raw_scores,
            anchors,
            min_score_thresh=min_score_thresh,
            min_suppression_threshold=0.3,
        )
        print(f"Detect  {detections.shape[0]} faces")
    except Exception as e:
        print(f"postprocess failed: {e}")
        import traceback

        traceback.print_exc()
        return

    if output_path is None:
        output_path = f"result_{os.path.basename(image_path)}"

    visualize_detections(img_resized, detections, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BlazeFace postprocess tool")
    parser.add_argument("--image", type=str, required=True, help="path for picture")
    parser.add_argument("--confidence", type=float, default=0.65, help="confidence threshold")

    boxes_file = "./model_zoo/BlazeFace_infer_int8/out_float/output_0.bin"
    scores_file = "./model_zoo/BlazeFace_infer_int8/out_float/output_1.bin"
    anchors_path = "./model_zoo/BlazeFace_infer_int8/anchorsback.npy"
    output_path = "./model_zoo/BlazeFace_infer_int8/detection_result.jpg"

    args = parser.parse_args()
    process_face_detection(
        boxes_file,
        scores_file,
        args.image,
        anchors_path,
        output_path,
        min_score_thresh=args.confidence,
    )
