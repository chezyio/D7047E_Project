import argparse
import os
import clearml
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.utils.file import save_json
import cv2
import glob
import torch
from ultralytics import YOLO

def to_yolo_bbox(voc_bbox, image_width, image_height):
    """
    Convert VOC bbox [x_min, y_min, x_max, y_max] to YOLO format [center_x, center_y, width, height].
    Normalized by image dimensions.
    """
    x_min, y_min, x_max, y_max = voc_bbox
    width = x_max - x_min
    height = y_max - y_min
    center_x = x_min + width / 2
    center_y = y_min + height / 2
    return [
        center_x / image_width,
        center_y / image_height,
        width / image_width,
        height / image_height
    ]

def main(opt):
    yolo_model_name: str = opt.yolo_model if ".pt" in opt.yolo_model else opt.yolo_model + ".pt"
    conf: float = opt.conf_thres
    project = opt.project
    run_name = os.path.basename(yolo_model_name) if opt.name is None else opt.name
    source = opt.source
    slice_size = opt.slice_size

    clearml.browser_login()

    # Configure output directory
    output_dir = os.path.join(project, run_name)
    os.makedirs(output_dir, exist_ok=True)

    # Initialize SAHI detection model
    if "yolov5" in yolo_model_name.lower():
        detection_model = AutoDetectionModel.from_pretrained(
            model_type="yolov5",
            model_path=yolo_model_name,
            confidence_threshold=conf,
            image_size=1920,
            device="cuda:0"  # Adjust to "cpu" if needed
        )
    elif "yolov8" in yolo_model_name.lower():
        detection_model = AutoDetectionModel.from_pretrained(
            model_type="yolov8",  # YOLOv11 uses YOLOv8 compatibility
            model_path=yolo_model_name,
            confidence_threshold=conf,
            image_size=1920,
            device="cuda:0"
        )
    else:
        raise ValueError(f"Unsupported model: {yolo_model_name}. Must contain 'yolov5' or 'yolo11'.")

    # Process source
    if os.path.isfile(source) and source.lower().endswith(('.mp4', '.avi', '.mov')):
        # Video processing
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {source}")

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Setup output video
        output_video_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(source))[0]}_detected.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            print(f"Processing frame {frame_count}/{total_frames}")

            # Perform sliced prediction
            result = get_sliced_prediction(
                image=frame,
                detection_model=detection_model,
                slice_height=slice_size,
                slice_width=slice_size,
                overlap_height_ratio=0.2,
                overlap_width_ratio=0.2,
                perform_standard_pred=False
            )

            # Draw predictions
            for object_prediction in result.object_prediction_list:
                bbox = object_prediction.bbox.to_voc_bbox()
                try:
                    x1, y1, x2, y2 = [int(coord) for coord in bbox]
                    label = object_prediction.category.name
                    score = object_prediction.score.value
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{label} {score:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                except (ValueError, TypeError) as e:
                    print(f"Skipping invalid bbox: {bbox}, error: {e}")

            # Save to video
            out.write(frame)

            # Save labels for this frame
            label_path = os.path.join(output_dir, f"frame_{frame_count:06d}.txt")
            with open(label_path, 'w') as f:
                for object_prediction in result.object_prediction_list:
                    bbox = object_prediction.bbox.to_voc_bbox()
                    try:
                        yolo_bbox = to_yolo_bbox(bbox, width, height)
                        label = object_prediction.category.id
                        score = object_prediction.score.value
                        f.write(f"{label} {yolo_bbox[0]:.6f} {yolo_bbox[1]:.6f} {yolo_bbox[2]:.6f} {yolo_bbox[3]:.6f} {score:.6f}\n")
                    except (ValueError, TypeError) as e:
                        print(f"Skipping invalid YOLO bbox: {bbox}, error: {e}")

            # Save COCO JSON for this frame
            result_json = result.to_coco_annotations()
            save_json(result_json, os.path.join(output_dir, f"frame_{frame_count:06d}_sahi.json"))

            # Clear GPU memory
            torch.cuda.empty_cache()

        cap.release()
        out.release()

    else:
        # Image processing (file or directory)
        if os.path.isfile(source):
            image_paths = [source]
        else:
            image_paths = glob.glob(os.path.join(source, "*.jpg")) + \
                          glob.glob(os.path.join(source, "*.png"))

        for image_path in image_paths:
            image = cv2.imread(image_path)
            if image is None:
                print(f"Cannot read image: {image_path}")
                continue

            # Perform sliced prediction
            result = get_sliced_prediction(
                image=image,
                detection_model=detection_model,
                slice_height=slice_size,
                slice_width=slice_size,
                overlap_height_ratio=0.2,
                overlap_width_ratio=0.2,
                perform_standard_pred=False
            )

            # Save results
            image_name = os.path.basename(image_path)
            output_image_path = os.path.join(output_dir, image_name)
            output_label_path = os.path.join(output_dir, f"{os.path.splitext(image_name)[0]}.txt")

            # Draw predictions
            for object_prediction in result.object_prediction_list:
                bbox = object_prediction.bbox.to_voc_bbox()
                try:
                    x1, y1, x2, y2 = [int(coord) for coord in bbox]
                    label = object_prediction.category.name
                    score = object_prediction.score.value
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(image, f"{label} {score:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                except (ValueError, TypeError) as e:
                    print(f"Skipping invalid bbox: {bbox}, error: {e}")

            cv2.imwrite(output_image_path, image)

            # Save labels
            with open(output_label_path, 'w') as f:
                for object_prediction in result.object_prediction_list:
                    bbox = object_prediction.bbox.to_voc_bbox()
                    try:
                        yolo_bbox = to_yolo_bbox(bbox, image.shape[1], image.shape[0])
                        label = object_prediction.category.id
                        score = object_prediction.score.value
                        f.write(f"{label} {yolo_bbox[0]:.6f} {yolo_bbox[1]:.6f} {yolo_bbox[2]:.6f} {yolo_bbox[3]:.6f} {score:.6f}\n")
                    except (ValueError, TypeError) as e:
                        print(f"Skipping invalid YOLO bbox: {bbox}, error: {e}")

            # Save COCO JSON
            result_json = result.to_coco_annotations()
            save_json(result_json, os.path.join(output_dir, f"{os.path.splitext(image_name)[0]}_sahi.json"))

            # Clear GPU memory
            torch.cuda.empty_cache()

def parse_opt(known=False):
    parser = argparse.ArgumentParser(
        description='This python script detects cars using a trained YOLO network with SAHI slicing.'
    )
    parser.add_argument('--yolo_model',
                        type=str,
                        help='YOLO model to use for detection (must contain "yolov5" or "yolo11").',
                        required=True)
    parser.add_argument('--source',
                        type=str,
                        help='file/dir containing images or a video file (.mp4, .avi, .mov)',
                        required=True)
    parser.add_argument('--conf_thres',
                        type=float,
                        help='confidence threshold (default = 0.25)',
                        default=0.25)
    parser.add_argument(
        '--project',
        type=str,
        help="Project name. If omitted, is set to 'runs/detect'.",
        default='runs/detect')
    parser.add_argument('--name',
                        type=str,
                        help='Task name. If omitted, yolo model name is used.',
                        default=None)
    parser.add_argument('--slice_size',
                        type=int,
                        help='Size of the sliced images for SAHI (default = 960)',
                        default=960)

    return parser.parse_known_args()[0] if known else parser.parse_args()

def run(**kwargs):
    opt = parse_opt(True)
    for k, v in kwargs.items():
        setattr(opt, k, v)
    main(opt)

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)