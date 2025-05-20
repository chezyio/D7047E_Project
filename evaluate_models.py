import os
import random
import logging
import numpy as np
from ultralytics import YOLO
import cv2
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def compute_iou(box1, box2, w, h):
    """compute IoU between two normalized bounding boxes."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    x1_min, y1_min = (x1 - w1 / 2) * w, (y1 - h1 / 2) * h
    x1_max, y1_max = (x1 + w1 / 2) * w, (y1 + h1 / 2) * h
    x2_min, y2_min = (x2 - w2 / 2) * w, (y2 - h2 / 2) * h
    x2_max, y2_max = (x2 + w2 / 2) * w, (y2 + h2 / 2) * h

    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)
    area1 = w1 * h1 * w * h
    area2 = w2 * h2 * w * h
    union_area = area1 + area2 - inter_area

    return inter_area / union_area if union_area > 0 else 0

def evaluate_predictions(pred_boxes, gt_boxes, w, h, iou_thresholds=[0.5], class_agnostic=False):
    """evaluate predictions with precision, recall, and F1."""
    metrics = defaultdict(list)

    for iou_thr in iou_thresholds:
        tp, fp, fn = 0, 0, len([b for b, _ in gt_boxes]) if not class_agnostic else len(gt_boxes)
        matched_gt = set()

        for pred_idx, (pred_box, pred_class, conf) in enumerate(pred_boxes):
            max_iou = 0
            max_gt_idx = -1
            for gt_idx, (gt_box, gt_class) in enumerate(gt_boxes):
                if gt_idx in matched_gt:
                    continue
                if not class_agnostic and pred_class != gt_class:
                    continue
                iou = compute_iou(pred_box, gt_box, w, h)
                if iou > max_iou:
                    max_iou = iou
                    max_gt_idx = gt_idx

            if max_iou >= iou_thr and max_gt_idx != -1:
                tp += 1
                fn -= 1
                matched_gt.add(max_gt_idx)
            else:
                fp += 1

        # compute precision, recall, F1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        metrics[iou_thr].append({"precision": precision, "recall": recall, "f1": f1})

    return metrics

def get_test_images(data_dir):
    """read test image paths from test.txt."""
    test_file = os.path.join(data_dir, "test.txt")
    if not os.path.exists(test_file):
        raise FileNotFoundError(f"Test file not found: {test_file}")
    with open(test_file, 'r') as f:
        test_images = [line.strip() for line in f if line.strip()]
    test_images = [os.path.join(data_dir, img) if not os.path.isabs(img) else img for img in test_images]
    valid_images = [img for img in test_images if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
    accessible_images = [img for img in valid_images if os.path.exists(img)]
    if not accessible_images:
        raise ValueError("No valid or accessible images found.")
    return accessible_images

def load_ground_truths(label_dir, image_path):
    """load ground truth bounding boxes and classes for an image."""
    label_file = os.path.join(label_dir, os.path.basename(image_path).replace('.jpg', '.txt').replace('.png', '.txt'))
    if not os.path.exists(label_file):
        logging.warning(f"Label file not found for {image_path}")
        return []
    with open(label_file, 'r') as f:
        boxes = []
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            class_id = int(parts[0])
            box = list(map(float, parts[1:5]))
            boxes.append((box, class_id))
    return boxes

def draw_boxes(image, boxes, color, label=""):
    """draw bounding boxes on an image with labels stacked in order: YOLO, SAHI, GT."""
    h, w, _ = image.shape
    for box, _ in boxes:
        cx, cy, bw, bh = box
        x1, y1 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
        x2, y2 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        if label:
            base_y = y1 - 10 
            if base_y < 20:
                base_y = y1 + 20
            
            if label == "YOLO":
                text_y = base_y
            elif label == "SAHI":
                text_y = base_y + 15
            elif label == "GT":
                text_y = base_y + 30
            else:
                text_y = base_y
            
            cv2.putText(image, label, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return image

def evaluate_models(models, data_dir, label_dir, save_dir="PROD_models", seed=42, num_samples=50, slice_size=320, iou_thresholds=[0.5, 0.75]):
    random.seed(seed)

    # load test images
    try:
        test_images = get_test_images(data_dir)
    except Exception as e:
        logging.error(f"Failed to load test images: {e}")
        return

    if len(test_images) < num_samples:
        logging.error(f"Not enough test images. Found {len(test_images)}, but need {num_samples}.")
        return

    # random subset
    selected_images = random.sample(test_images, num_samples)
    logging.info(f"Selected {num_samples} images for evaluation: {selected_images}")

    # summary file for metrics
    summary_file = os.path.join(save_dir, "evaluation_summary.txt")
    with open(summary_file, 'w') as sf:
        sf.write("Model Evaluation Summary\n")
        sf.write("=" * 50 + "\n")

    # evaluate each model
    for model_path in models:
        logging.info(f"Evaluating model: {model_path}")
        if not os.path.exists(model_path):
            logging.error(f"Model file not found: {model_path}")
            continue

        try:
            model = YOLO(model_path)
            model_save_name = os.path.basename(model_path).replace('.pt', '_results')
            output_dir = os.path.join(save_dir, model_save_name)
            os.makedirs(output_dir, exist_ok=True)

            # initialize metrics
            yolo_metrics = defaultdict(list)
            sahi_metrics = defaultdict(list)

            logging.info(f"Running standard YOLO inference for {model_path}")

            # YOLO inference
            results = model.predict(
                source=selected_images,
                conf=0.9,
                iou=0.45,
                verbose=True
            )

            # SAHI inference
            logging.info(f"Running SAHI sliced inference for {model_path}")
            sahi_model = AutoDetectionModel.from_pretrained(
                model_type="yolov8",
                model_path=model_path,
                confidence_threshold=0.9,
                device="cuda:0" if model.device.type == "cuda" else "cpu"
            )

            for i, image_path in enumerate(selected_images):
                # load image and ground truth boxes
                image_yolo = cv2.imread(image_path)
                h, w, _ = image_yolo.shape
                ground_truth_boxes = load_ground_truths(label_dir, image_path)

                # YOLO evaluation
                result = results[i]
                predicted_boxes = []
                if result.boxes:
                    for box, cls, conf in zip(result.boxes.xywhn.tolist(), result.boxes.cls.tolist(), result.boxes.conf.tolist()):
                        predicted_boxes.append((box, int(cls), conf))
                yolo_metrics_i = evaluate_predictions(predicted_boxes, ground_truth_boxes, w, h, iou_thresholds)
                for iou_thr in iou_thresholds:
                    yolo_metrics[iou_thr].append(yolo_metrics_i[iou_thr][0])

                # YOLO comparison image
                image_yolo = draw_boxes(image_yolo, ground_truth_boxes, color=(0, 255, 0), label="GT")
                image_yolo = draw_boxes(image_yolo, [(b, None) for b, _, _ in predicted_boxes], color=(0, 0, 255), label="YOLO")
                yolo_output_path = os.path.join(output_dir, f"yolo_{os.path.basename(image_path)}")
                cv2.imwrite(yolo_output_path, image_yolo)
                logging.info(f"Saved YOLO comparison image to: {yolo_output_path}")

                # SAHI evaluation
                image_sahi = cv2.imread(image_path)
                sahi_result = get_sliced_prediction(
                    image_path,
                    sahi_model,
                    slice_height=slice_size,
                    slice_width=slice_size,
                    overlap_height_ratio=0.2,
                    overlap_width_ratio=0.2
                )
                sahi_boxes = []
                for obj in sahi_result.object_prediction_list:
                    bbox = obj.bbox
                    cx = (bbox.minx + bbox.maxx) / 2 / w
                    cy = (bbox.miny + bbox.maxy) / 2 / h
                    bw = (bbox.maxx - bbox.minx) / w
                    bh = (bbox.maxy - bbox.miny) / h
                    class_id = obj.category.id if obj.category else 0
                    conf = obj.score.value if obj.score else 0.9
                    sahi_boxes.append(([cx, cy, bw, bh], class_id, conf))
                sahi_metrics_i = evaluate_predictions(sahi_boxes, ground_truth_boxes, w, h, iou_thresholds)
                for iou_thr in iou_thresholds:
                    sahi_metrics[iou_thr].append(sahi_metrics_i[iou_thr][0])

                # SAHI comparison image
                image_sahi = draw_boxes(image_sahi, ground_truth_boxes, color=(0, 255, 0), label="GT")
                image_sahi = draw_boxes(image_sahi, [(b, None) for b, _, _ in sahi_boxes], color=(255, 0, 0), label="SAHI")
                sahi_output_path = os.path.join(output_dir, f"sahi_{os.path.basename(image_path)}")
                cv2.imwrite(sahi_output_path, image_sahi)
                logging.info(f"Saved SAHI comparison image to: {sahi_output_path}")

                # combined comparison image
                image_combined = cv2.imread(image_path)
                image_combined = draw_boxes(image_combined, ground_truth_boxes, color=(0, 255, 0), label="GT")
                image_combined = draw_boxes(image_combined, [(b, None) for b, _, _ in predicted_boxes], color=(0, 0, 255), label="YOLO")
                image_combined = draw_boxes(image_combined, [(b, None) for b, _, _ in sahi_boxes], color=(255, 0, 0), label="SAHI")
                combined_output_path = os.path.join(output_dir, f"combined_{os.path.basename(image_path)}")
                cv2.imwrite(combined_output_path, image_combined)
                logging.info(f"Saved combined comparison image to: {combined_output_path}")

            # compute average metrics
            metrics_summary = f"\nModel: {model_path}\n"
            for iou_thr in iou_thresholds:
                yolo_avg = {k: np.mean([m[k] for m in yolo_metrics[iou_thr]]) if yolo_metrics[iou_thr] else 0 for k in ["precision", "recall", "f1"]}
                sahi_avg = {k: np.mean([m[k] for m in sahi_metrics[iou_thr]]) if sahi_metrics[iou_thr] else 0 for k in ["precision", "recall", "f1"]}
                
                metrics_summary += (
                    f"\nIoU Threshold: {iou_thr}\n"
                    f"YOLO Metrics:\n"
                    f"  Mean Precision: {yolo_avg['precision']:.4f}\n"
                    f"  Mean Recall: {yolo_avg['recall']:.4f}\n"
                    f"  Mean F1-Score: {yolo_avg['f1']:.4f}\n"
                    f"SAHI Metrics:\n"
                    f"  Mean Precision: {sahi_avg['precision']:.4f}\n"
                    f"  Mean Recall: {sahi_avg['recall']:.4f}\n"
                    f"  Mean F1-Score: {sahi_avg['f1']:.4f}\n"
                )

            metrics_summary += "=" * 50 + "\n"

            logging.info(metrics_summary)
            with open(summary_file, 'a') as sf:
                sf.write(metrics_summary)

        except Exception as e:
            logging.error(f"Error evaluating model {model_path}: {e}")

if __name__ == "__main__":
    models = ["PROD_models/yolov8n.pt", "PROD_models/yolov9t.pt", "PROD_models/yolov10n.pt", "PROD_models/yolo11n.pt"]
    data_dir = "./data"
    label_dir = "./data/labels"
    evaluate_models(models, data_dir, label_dir, slice_size=960, iou_thresholds=[0.5, 0.75])