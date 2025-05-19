# import argparse
# import os
# from detectron2.config import get_cfg
# from detectron2 import model_zoo
# from detectron2.engine import DefaultPredictor
# from detectron2.utils.visualizer import Visualizer
# import cv2
# from tqdm import tqdm

# def create_detectron2_config():
#     cfg = get_cfg()
#     cfg.MODEL.DEVICE = 'cpu'
#     cfg.merge_from_file(
#     model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
#     cfg.DATASETS.TRAIN = ("train_set", )
#     cfg.TEST.EVAL_PERIOD = 1
#     cfg.DATALOADER.NUM_WORKERS = 4
#     cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
#     cfg.SOLVER.IMS_PER_BATCH = 16  # This is the real "batch size" commonly known to deep learning people
#     cfg.SOLVER.BASE_LR = 0.001  # pick a good LR
#     cfg.SOLVER.MAX_ITER = 300  # 300 iterations seems good
#     cfg.SOLVER.WARMUP_ITERS = 200
#     cfg.SOLVER.STEPS = []  # do not decay learning rate
#     cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  # The "RoIHead batch size". 128 is faster, and good (default: 512)
#     cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (car). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
#     # NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.

#     ##### Modified by me ###########
#     # cfg.MODEL.ROI_MASK_HEAD.NAME = None
#     # cfg.MODEL.ROI_KEYPOINT_HEAD.NAME = None
#     # cfg.MODEL.SEM_SEG_HEAD.NAME = None
#     cfg.DATASETS.TEST = ("val_set", )
#     cfg.SOLVER.REFERENCE_WORLD_SIZE = 0
#     cfg.SOLVER.GAMMA = 0.05
#     cfg.TEST.EVAL_PERIOD = 50
#     cfg.INPUT.MIN_SIZE_TEST = 1080
#     cfg.INPUT.MIN_SIZE_TRAIN = (920, 952, 984, 1016, 1048, 1080)
#     cfg.INPUT.MAX_SIZE_TEST = 1920
#     cfg.INPUT.MAX_SIZE_TRAIN = 1920
#     return cfg

# def detect(img_file_name: str, out_dir, predictor: DefaultPredictor):
#     im = cv2.imread(img_file_name)
#     outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
#     v = Visualizer(im[:, :, ::-1])
#     out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
#     f_name = os.path.join(out_dir, os.path.basename(img_file_name))
#     cv2.imwrite(f_name, out.get_image()[:, :, ::-1])
    
# def main(opt):
#     run_name = os.path.basename(
#         opt.model) if opt.name is None else opt.name
#     out_dir = os.path.join(opt.project, run_name)
#     os.makedirs(out_dir, exist_ok=True)
    
#     cfg = create_detectron2_config()
#     cfg.MODEL.WEIGHTS = opt.model
#     cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = opt.conf_thres
#     predictor = DefaultPredictor(cfg)
    
#     if os.path.isfile(opt.source):
#         detect(opt.source, out_dir, predictor)
#     elif os.path.isdir(opt.source):
#         for filename in tqdm(os.listdir(opt.source)):
#             file_path = os.path.join(opt.source, filename)
#             if os.path.isfile(file_path):
#                 detect(file_path, out_dir, predictor)
#     else:
#         print(f"{opt.source} is neither a file nor a directory.")

# def parse_opt(known=False):
#     parser = argparse.ArgumentParser(
#         description=
#         'This python script detect cars using a trained Faster R-CNN network.')
#     parser.add_argument('--model',
#                         type=str,
#                         help='Faster R-CNN model to use for detection.',
#                         required=True)
#     parser.add_argument('--source', type=str, help='file/dir', required=True)
#     parser.add_argument('--conf_thres',
#                         type=float,
#                         help='confidence threshold',
#                         default=0.25)
#     parser.add_argument(
#         '--project',
#         type=str,
#         help="Project name. If omitted, is set to 'runs/detect'.",
#         default='runs/detect')
#     parser.add_argument('--name',
#                         type=str,
#                         help='Task name. If omitted, the model name is used.',
#                         default=None)

#     return parser.parse_known_args()[0] if known else parser.parse_args()

# def run(**kwargs):
#     opt = parse_opt(True)
#     for k, v in kwargs.items():
#         setattr(opt, k, v)
#     main(opt)


# if __name__ == "__main__":
#     opt = parse_opt()
#     main(opt)


import argparse
import os
import glob
import logging
import cv2
from tqdm import tqdm
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog
from detectron2.data.datasets import register_coco_instances

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def create_detectron2_config(model_path, conf_thres, device='cpu'):
    cfg = get_cfg()
    cfg.MODEL.DEVICE = device
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = conf_thres
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # car
    cfg.INPUT.MIN_SIZE_TEST = 768
    cfg.INPUT.MAX_SIZE_TEST = 1280
    return cfg

def register_dataset(data_dir):
    """Register dataset for metadata (optional, for visualization)."""
    try:
        coco_json = os.path.join(data_dir, "val", "_annotations.coco.json")
        if os.path.exists(coco_json):
            register_coco_instances(
                "my_dataset_val",
                {},
                coco_json,
                os.path.join(data_dir, "val")
            )
            MetadataCatalog.get("my_dataset_val").set(thing_classes=["car"])
            logging.info("Dataset metadata registered successfully")
            return MetadataCatalog.get("my_dataset_val")
        else:
            logging.warning("COCO annotations not found; using default metadata")
            return None
    except Exception as e:
        logging.warning(f"Failed to register dataset: {e}. Using default metadata.")
        return None

def get_test_images(data_dir, height_based=False):
    """Read test image paths from test.txt or height-based test files."""
    test_images = []
    if height_based:
        height_test_files = glob.glob(os.path.join(data_dir, "height*-test.txt"))
        if not height_test_files:
            raise FileNotFoundError("No height-based test files found in data_dir")
        for test_file in height_test_files:
            try:
                with open(test_file, 'r') as f:
                    images = [line.strip() for line in f if line.strip()]
                # Convert relative paths to absolute
                images = [os.path.join(data_dir, img) if not os.path.isabs(img) else img for img in images]
                test_images.extend(images)
                logging.info(f"Loaded {len(images)} images from {test_file}")
            except Exception as e:
                logging.error(f"Failed to read {test_file}: {e}")
    else:
        test_file = os.path.join(data_dir, "test.txt")
        if not os.path.exists(test_file):
            raise FileNotFoundError(f"Test file not found: {test_file}")
        try:
            with open(test_file, 'r') as f:
                test_images = [line.strip() for line in f if line.strip()]
            # Convert relative paths to absolute
            test_images = [os.path.join(data_dir, img) if not os.path.isabs(img) else img for img in test_images]
            logging.info(f"Loaded {len(test_images)} images from {test_file}")
        except Exception as e:
            logging.error(f"Failed to read {test_file}: {e}")
            raise
    # Filter for valid image extensions
    test_images = [img for img in test_images if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
    return test_images

def detect(img_file_name: str, out_dir, predictor: DefaultPredictor, metadata):
    """Run inference on a single image and save visualization."""
    try:
        im = cv2.imread(img_file_name)
        if im is None:
            logging.error(f"Failed to load image: {img_file_name}")
            return
        outputs = predictor(im)
        logging.info(f"Predictor output for {img_file_name}: {outputs}")
        if "instances" in outputs and len(outputs["instances"]) > 0:
            v = Visualizer(im[:, :, ::-1], metadata=metadata, scale=0.5, instance_mode=ColorMode.IMAGE)
            out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            f_name = os.path.join(out_dir, f"pred_{os.path.basename(img_file_name)}")
            cv2.imwrite(f_name, out.get_image()[:, :, ::-1])
            logging.info(f"Saved prediction visualization to {f_name}")
        else:
            logging.warning(f"No instances found in output for {img_file_name}")
    except Exception as e:
        logging.error(f"Failed to process {img_file_name}: {e}")

def main(opt):
    run_name = os.path.basename(opt.model) if opt.name is None else opt.name
    out_dir = os.path.join(opt.project, run_name)
    os.makedirs(out_dir, exist_ok=True)

    # Register dataset for metadata
    metadata = register_dataset(opt.data_dir)

    # Create predictor
    cfg = create_detectron2_config(opt.model, opt.conf_thres, device='cpu')
    try:
        predictor = DefaultPredictor(cfg)
        logging.info("Predictor initialized successfully")
    except Exception as e:
        logging.error(f"Failed to initialize predictor: {e}")
        return

    # Load test images
    try:
        test_images = get_test_images(opt.data_dir, opt.height_based)
        if not test_images:
            logging.error("No valid test images found")
            return
    except Exception as e:
        logging.error(f"Failed to load test images: {e}")
        return

    # Process test images
    for img_path in tqdm(test_images, desc="Processing test images"):
        if os.path.isfile(img_path):
            detect(img_path, out_dir, predictor, metadata)
        else:
            logging.warning(f"Skipping invalid path: {img_path}")

def parse_opt(known=False):
    parser = argparse.ArgumentParser(description='Detect cars using a trained Faster R-CNN model on prepared test data.')
    parser.add_argument('--model', type=str, help='Path to Faster R-CNN model weights (.pth)', required=True)
    parser.add_argument('--data_dir', type=str, help='Path to prepared data directory (containing images/, labels/, test.txt)', required=True)
    parser.add_argument('--conf_thres', type=float, default=0.05, help='Confidence threshold for predictions')
    parser.add_argument('--project', type=str, default='runs/detect', help='Project directory for outputs')
    parser.add_argument('--name', type=str, default=None, help='Task name (defaults to model name)')
    parser.add_argument('--height_based', action='store_true', help='Use height-based test files (e.g., height(0, 175)-test.txt)')
    return parser.parse_known_args()[0] if known else parser.parse_args()

def run(**kwargs):
    opt = parse_opt(True)
    for k, v in kwargs.items():
        setattr(opt, k, v)
    main(opt)

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)