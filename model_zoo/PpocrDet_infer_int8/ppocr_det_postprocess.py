import cv2
import numpy as np
from pathlib import Path
import torch


def get_mini_boxes(contour):
    """
    Get the minimum enclosing rectangle
    """
    bounding_box = cv2.minAreaRect(contour)
    points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

    index_1, index_2, index_3, index_4 = 0, 1, 2, 3
    if points[3][1] <= points[2][1]:
        index_2 = 3
        index_3 = 2
    else:
        index_2 = 2
        index_3 = 3
    if points[1][1] <= points[0][1]:
        index_1 = 1
        index_4 = 0
    else:
        index_1 = 0
        index_4 = 1

    box = [points[index_1], points[index_2], points[index_3], points[index_4]]
    return box, min(bounding_box[1])


def get_contour_area(box, unclip_ratio):
    """
    Calculate the distance for Unclip
    """
    box = np.array(box)
    pts_num = 4
    area = 0.0
    dist = 0.0

    for i in range(pts_num):
        area += (
            box[i][0] * box[(i + 1) % pts_num][1]
            - box[i][1] * box[(i + 1) % pts_num][0]
        )
        dist += np.sqrt(
            (box[i][0] - box[(i + 1) % pts_num][0]) ** 2
            + (box[i][1] - box[(i + 1) % pts_num][1]) ** 2
        )

    area = abs(area / 2.0)
    distance = area * unclip_ratio / dist
    return distance


def unclip(box, unclip_ratio):
    """
    Implement box expansion using the Clipper library concept
    """
    from shapely.geometry import Polygon

    poly = Polygon(box)
    distance = get_contour_area(box, unclip_ratio)
    expanded = poly.buffer(distance)

    # Get the coordinates after expansion
    if expanded.is_empty:
        return None

    points = np.array(expanded.exterior.coords)
    # Calculate the minimum enclosing rectangle
    rect = cv2.minAreaRect(points.astype(np.float32))
    return rect


def box_score_fast(box_array, pred):
    """
    Calculate the score inside the box
    """
    h, w = pred.shape[:2]
    box = np.array(box_array)

    xmin = np.clip(np.floor(box[:, 0].min()).astype(np.int32), 0, w - 1)
    xmax = np.clip(np.ceil(box[:, 0].max()).astype(np.int32), 0, w - 1)
    ymin = np.clip(np.floor(box[:, 1].min()).astype(np.int32), 0, h - 1)
    ymax = np.clip(np.ceil(box[:, 1].max()).astype(np.int32), 0, h - 1)

    mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
    box[:, 0] = box[:, 0] - xmin
    box[:, 1] = box[:, 1] - ymin
    cv2.fillPoly(mask, [box.astype(np.int32)], 1)

    score = cv2.mean(pred[ymin : ymax + 1, xmin : xmax + 1], mask)[0]
    return score


def polygon_score_acc(contour, pred):
    """
    Calculate the polygon score
    """
    h, w = pred.shape[:2]
    contour = np.array(contour)

    xmin = np.clip(np.floor(contour[:, 0].min()).astype(np.int32), 0, w - 1)
    xmax = np.clip(np.ceil(contour[:, 0].max()).astype(np.int32), 0, w - 1)
    ymin = np.clip(np.floor(contour[:, 1].min()).astype(np.int32), 0, h - 1)
    ymax = np.clip(np.ceil(contour[:, 1].max()).astype(np.int32), 0, h - 1)

    mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
    contour[:, 0] = contour[:, 0] - xmin
    contour[:, 1] = contour[:, 1] - ymin
    cv2.fillPoly(mask, [contour.astype(np.int32)], 1)

    score = cv2.mean(pred[ymin : ymax + 1, xmin : xmax + 1], mask)[0]
    return score


def boxes_from_bitmap(pred, bitmap, config):
    """
    Get detection boxes from binary image
    """
    min_size = 3
    max_candidates = 1000
    box_thresh = float(config.get("det_db_box_thresh", 0.3))
    unclip_ratio = float(config.get("det_db_unclip_ratio", 1.5))
    use_polygon_score = int(config.get("det_use_polygon_score", 0))

    height, width = bitmap.shape

    # Find contours
    contours, _ = cv2.findContours(bitmap, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Limit the number of contours
    num_contours = min(len(contours), max_candidates)

    boxes = []

    for i in range(num_contours):
        contour = contours[i]

        # Skip contours that are too small
        if len(contour) <= 2:
            continue

        # Get the minimum enclosing rectangle
        box, ssid = get_mini_boxes(contour)

        if ssid < min_size:
            continue

        # Calculate the score
        if use_polygon_score:
            score = polygon_score_acc(contour, pred)
        else:
            score = box_score_fast(box, pred)

        if score < box_thresh:
            continue

        # Unclip to expand the detection box
        rect = unclip(box, unclip_ratio)
        if rect is None:
            continue

        # Get the expanded box
        clip_box, ssid = get_mini_boxes(cv2.boxPoints(rect).astype(np.int32))

        if ssid < min_size + 2:
            continue

        # Map the coordinates back to the original size
        dest_width = pred.shape[1]
        dest_height = pred.shape[0]
        intcliparray = []

        for point in clip_box:
            x = int(np.clip(np.round(point[0] / width * dest_width), 0, dest_width))
            y = int(np.clip(np.round(point[1] / height * dest_height), 0, dest_height))
            intcliparray.append([x, y])

        boxes.append(intcliparray)

    return boxes


def order_points_clockwise(pts):
    """
    Rearrange points in a clockwise order
    """
    pts = np.array(pts)

    # Sort by x-coordinate
    xsorted = pts[np.argsort(pts[:, 0])]

    # Two points on the left and two points on the right
    leftmost = xsorted[:2]
    rightmost = xsorted[2:]

    # Sort left and right points by y-coordinate
    leftmost = leftmost[np.argsort(leftmost[:, 1])]
    rightmost = rightmost[np.argsort(rightmost[:, 1])]

    # Return points in clockwise order: top-left, top-right, bottom-right, bottom-left
    rect = np.array([leftmost[0], rightmost[0], rightmost[1], leftmost[1]])
    return rect.tolist()


def filter_tag_det_res(boxes, ratio_h, ratio_w, srcimg):
    """
    Filter detection results and map coordinates back to the original image size
    """
    oriimg_h, oriimg_w = srcimg.shape[:2]

    root_points = []
    for box in boxes:
        # Sort points in clockwise order
        box = order_points_clockwise(box)

        # Map coordinates back to the original image size
        for i in range(len(box)):
            box[i][0] = int(box[i][0] / ratio_w)
            box[i][1] = int(box[i][1] / ratio_h)

            # Ensure coordinates are within the image bounds
            box[i][0] = max(0, min(box[i][0], oriimg_w - 1))
            box[i][1] = max(0, min(box[i][1], oriimg_h - 1))

        # Filter out boxes that are too small
        rect_width = int(
            np.sqrt((box[0][0] - box[1][0]) ** 2 + (box[0][1] - box[1][1]) ** 2)
        )
        rect_height = int(
            np.sqrt((box[0][0] - box[3][0]) ** 2 + (box[0][1] - box[3][1]) ** 2)
        )

        if rect_width <= 4 or rect_height <= 4:
            continue

        root_points.append(box)

    return root_points


def post_process_model_output(model_output, original_img, config):
    """
    Post-process the model output

    Args:
        model_output: Model output, shape [1, 1, h, w]
        original_img: Original image in OpenCV BGR format
        config: Configuration dictionary

    Returns:
        dt_boxes: List of detection boxes
    """
    # Process model output to extract the probability map
    pred = model_output[0, 0]  # Extract probability map, shape [h, w]

    # Convert output to a binary map
    threshold = config.get("det_db_thresh", 0.3) * 255
    bitmap = np.zeros_like(pred, dtype=np.uint8)
    bitmap[pred * 255 >= threshold] = 255

    # Perform dilation if needed
    if config.get("det_db_use_dilate", 1) == 1:
        dilation_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        bitmap = cv2.dilate(bitmap, dilation_kernel)

    # Get detection boxes from the binary map
    boxes = boxes_from_bitmap(pred, bitmap, config)

    # Calculate scaling ratios
    h_ratio = pred.shape[0] / original_img.shape[0]
    w_ratio = pred.shape[1] / original_img.shape[1]

    # Filter detection results and map coordinates back to the original image size
    dt_boxes = filter_tag_det_res(boxes, h_ratio, w_ratio, original_img)

    return dt_boxes


def draw_text_det_res(dt_boxes, img_path, save_path):
    """
    Draw text detection results

    Args:
        dt_boxes: List of detection boxes
        img_path: Path to the original image
        save_path: Path to save the result

    Returns:
        None
    """
    src_im = cv2.imread(img_path)
    for box in dt_boxes:
        box = np.array(box).astype(np.int32).reshape(-1, 2)
        cv2.polylines(src_im, [box], True, color=(0, 255, 0), thickness=2)
    cv2.imwrite(save_path, src_im)
    print(f"Results saved to  {save_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Text Detection Post-Processing")
    parser.add_argument("--image", type=str, required=True, help="Path to the input image")
    parser.add_argument("--confidence", type=str, required=True, help="Confidence threshold")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the output image")
    parser.add_argument(
        "--model_output_path", type=str, required=True, help="Path to the model output binary file"
    )
    parser.add_argument(
        "--shape",
        type=str,
        default="1,1,640,640",
        help='Shape of the model output, format "n,c,h,w"',
    )
    parser.add_argument(
        "--do_conv",
        type=bool,
        default=True,
    )
    args = parser.parse_args()

    original_img = cv2.imread(args.image)

    model_output = np.fromfile(args.model_output_path, dtype=np.int8)
    model_output = model_output *0.032675705556794415
    model_output = model_output.astype(np.float32)

    if args.do_conv:
        # Convert model output to PyTorch tensor
        # model_output = torch.from_numpy(model_output).reshape(1, 24, 160, 160)
        model_output = torch.from_numpy(model_output).reshape(1, 160, 160, 24)
        model_output = model_output.permute(0, 3, 1, 2)
        conv1 = torch.nn.ConvTranspose2d(24, 24, 2, 2)
        conv1_weight = np.load("./model_zoo/PpocrDet_infer_int8/conv1.npy")
        conv1.weight = torch.nn.Parameter(torch.from_numpy(conv1_weight))
        conv1_bias = np.load("./model_zoo/PpocrDet_infer_int8/bias1.npy")
        conv1.bias = torch.nn.Parameter(torch.from_numpy(conv1_bias))
        conv2 = torch.nn.ConvTranspose2d(24, 1, 2, 2)
        conv2_weight = np.load("./model_zoo/PpocrDet_infer_int8/conv2.npy")
        conv2.weight = torch.nn.Parameter(torch.from_numpy(conv2_weight))
        conv2_bias = np.load("./model_zoo/PpocrDet_infer_int8/bias2.npy")
        conv2.bias = torch.nn.Parameter(torch.from_numpy(conv2_bias))
        model_output = torch.relu(model_output)
        model_output = conv1(model_output)
        model_output = torch.relu(model_output)
        model_output = conv2(model_output)
        model_output = torch.sigmoid(model_output)
        model_output = model_output.detach().numpy()
    # Reshape the array based on the specified shape
    shape = [int(dim) for dim in args.shape.split(",")]
    model_output = model_output.reshape(shape)

    config = {
        "det_db_thresh": 0.3,  # Binarization threshold
        "det_db_box_thresh": args.confidence,  # Detection box threshold
        "max_side_len": 960,  # Maximum side length
        "det_db_unclip_ratio": 3,  # Expansion ratio for detection boxes
        "det_db_min_size": 3.0,  # Minimum detection size
        "det_db_use_dilate": 1,  # Whether to use dilation
        "det_use_polygon_score": 0,  # Whether to use polygon score
    }
    # Perform post-processing
    dt_boxes = post_process_model_output(model_output, original_img, config)

    # Draw detection results and save
    draw_text_det_res(dt_boxes, args.image, args.output_path)
