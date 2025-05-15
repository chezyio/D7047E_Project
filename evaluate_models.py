import os
import random
import logging
from ultralytics import YOLO
import cv2
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_test_images(data_dir):
    """Read test image paths from test.txt."""
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
    """Load ground truth bounding boxes for an image."""
    label_file = os.path.join(label_dir, os.path.basename(image_path).replace('.jpg', '.txt').replace('.png', '.txt'))
    if not os.path.exists(label_file):
        logging.warning(f"Label file not found for {image_path}")
        return []
    with open(label_file, 'r') as f:
        boxes = [list(map(float, line.strip().split()[1:])) for line in f if line.strip()]
    return boxes

def draw_boxes(image, boxes, color, label=""):
    """Draw bounding boxes on an image with labels stacked in order: YOLO, SAHI, GT."""
    h, w, _ = image.shape
    for box in boxes:
        cx, cy, bw, bh = box
        x1, y1 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
        x2, y2 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        if label:
            # Base position for the topmost label (YOLO)
            base_y = y1 -10 
            if base_y < 20:  # Prevent labels from going off the top
                base_y = y1 + 20
            
            # Assign vertical positions based on label type
            if label == "YOLO":
                text_y = base_y  # Topmost position
            elif label == "SAHI":
                text_y = base_y + 15  # Below YOLO
            elif label == "GT":
                text_y = base_y + 30  # Below SAHI
            else:
                text_y = base_y  # Default case
            
            # Draw the label
            cv2.putText(image, label, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return image

def evaluate_models(models, data_dir, label_dir, save_dir="PROD_models", seed=42, num_samples=50, slice_size=320):
    random.seed(seed)

    # Load test images
    try:
        test_images = get_test_images(data_dir)
    except Exception as e:
        logging.error(f"Failed to load test images: {e}")
        return

    if len(test_images) < num_samples:
        logging.error(f"Not enough test images. Found {len(test_images)}, but need {num_samples}.")
        return

    # Random subset
    selected_images = random.sample(test_images, num_samples)
    logging.info(f"Selected {num_samples} images for evaluation: {selected_images}")

    # Evaluate each model
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
                # Load ground truth boxes
                ground_truth_boxes = load_ground_truths(label_dir, image_path)

                # YOLO comparison image
                image_yolo = cv2.imread(image_path)
                h, w, _ = image_yolo.shape
                image_yolo = draw_boxes(image_yolo, ground_truth_boxes, color=(0, 255, 0), label="GT")
                result = results[i]
                predicted_boxes = result.boxes.xywhn.tolist() if result.boxes else []
                image_yolo = draw_boxes(image_yolo, predicted_boxes, color=(0, 0, 255), label="YOLO")
                yolo_output_path = os.path.join(output_dir, f"yolo_{os.path.basename(image_path)}")
                cv2.imwrite(yolo_output_path, image_yolo)
                logging.info(f"Saved YOLO comparison image to: {yolo_output_path}")

                # SAHI comparison image
                image_sahi = cv2.imread(image_path)
                image_sahi = draw_boxes(image_sahi, ground_truth_boxes, color=(0, 255, 0), label="GT")
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
                    sahi_boxes.append([cx, cy, bw, bh])
                image_sahi = draw_boxes(image_sahi, sahi_boxes, color=(255, 0, 0), label="SAHI")
                sahi_output_path = os.path.join(output_dir, f"sahi_{os.path.basename(image_path)}")
                cv2.imwrite(sahi_output_path, image_sahi)
                logging.info(f"Saved SAHI comparison image to: {sahi_output_path}")

                # Combined comparison image
                image_combined = cv2.imread(image_path)
                image_combined = draw_boxes(image_combined, ground_truth_boxes, color=(0, 255, 0), label="GT")
                image_combined = draw_boxes(image_combined, predicted_boxes, color=(0, 0, 255), label="YOLO")
                image_combined = draw_boxes(image_combined, sahi_boxes, color=(255, 0, 0), label="SAHI")
                combined_output_path = os.path.join(output_dir, f"combined_{os.path.basename(image_path)}")
                cv2.imwrite(combined_output_path, image_combined)
                logging.info(f"Saved combined comparison image to: {combined_output_path}")

        except Exception as e:
            logging.error(f"Error evaluating model {model_path}: {e}")

models = ["PROD_models/yolov8n.pt", "PROD_models/yolov9t.pt", "PROD_models/yolov10n.pt", "PROD_models/yolo11n.pt"]
data_dir = "./data" 
label_dir = "./data/labels"

evaluate_models(models, data_dir, label_dir, slice_size=960)
