# Methods for prediction for this competition
from math import ceil
from typing import Any, Dict, List
from flag import Flags
from pathlib import Path
import cv2
import detectron2
import numpy as np
from numpy import ndarray
import pandas as pd
import torch
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.engine import DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.structures import BoxMode
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import ColorMode, Visualizer
from tqdm import tqdm
from utils import load_yaml
from config import thing_classes, category_name_to_id

inputdir = Path("./")
traineddir = inputdir / "results/v9"
flags: Flags = Flags().update(load_yaml(str(traineddir / "flags.yaml")))
debug = flags.debug
outdir = Path(flags.outdir)
cfg = get_cfg()
cfg.OUTPUT_DIR = str(outdir)
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.DATALOADER.NUM_WORKERS = 2
# Let training initialize from model zoo
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = flags.base_lr  # pick a good LR
cfg.SOLVER.MAX_ITER = flags.iter
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = flags.roi_batch_size_per_image
cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(thing_classes)
# NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.


cfg.MODEL.WEIGHTS = str(traineddir/"model_final.pth")
print("Original thresh", cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST)  # 0.05
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set a custom testing threshold
print("Changed  thresh", cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST)
predictor = DefaultPredictor(cfg)


def format_pred(labels: ndarray, boxes: ndarray, scores: ndarray) -> str:
    pred_strings = []
    for label, score, bbox in zip(labels, scores, boxes):
        xmin, ymin, xmax, ymax = bbox.astype(np.int64)
        pred_strings.append(f"{label} {score} {xmin} {ymin} {xmax} {ymax}")
    return " ".join(pred_strings)


def predict_batch(predictor: DefaultPredictor, im_list: List[ndarray]) -> List:
    with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
        inputs_list = []
        for original_image in im_list:
            # Apply pre-processing to image.
            if predictor.input_format == "RGB":
                # whether the model expects BGR inputs or RGB
                original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]
            # Do not apply original augmentation, which is resize.
            # image = predictor.aug.get_transform(original_image).apply_image(original_image)
            image = original_image
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
            inputs = {"image": image, "height": height, "width": width}
            inputs_list.append(inputs)
        predictions = predictor.model(inputs_list)
        return predictions


if __name__ == '__main__':
    # inputdir = Path("./")
    # traineddir = inputdir / "results/v9"
    # flags: Flags = Flags().update(load_yaml(str(traineddir / "flags.yaml")))
    # debug = flags.debug
    # outdir = Path(flags.outdir)
    # cfg = get_cfg()
    # cfg.OUTPUT_DIR = str(outdir)
    # cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    # cfg.DATALOADER.NUM_WORKERS = 2
    # # Let training initialize from model zoo
    # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    # cfg.SOLVER.IMS_PER_BATCH = 2
    # cfg.SOLVER.BASE_LR = flags.base_lr  # pick a good LR
    # cfg.SOLVER.MAX_ITER = flags.iter
    # cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = flags.roi_batch_size_per_image
    # cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(thing_classes)
    # # NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.
    

    # cfg.MODEL.WEIGHTS = str(traineddir/"model_final.pth")
    # print("Original thresh", cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST)  # 0.05
    # cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.0  # set a custom testing threshold
    # print("Changed  thresh", cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST)
    # predictor = DefaultPredictor(cfg)

    # original_image = "vinbigdata-chest-xray-resized-png-256x256/test/002a34c58c5b758217ed1f584ccbcfe9.png"
    original_image = cv2.imread("vinbigdata-chest-xray-resized-png-256x256/test/002a34c58c5b758217ed1f584ccbcfe9.png")
    # original_image = cv2.imread("00000001_000-Cardiomegaly.png")
    MetadataCatalog.get("vinbigdata_test").set(thing_classes=thing_classes)
    metadata = MetadataCatalog.get("vinbigdata_test")
    with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
        inputs_list = []
        if predictor.input_format == "RGB":
            print("rgb")
            original_image = original_image[:, :, ::-1]

        height, width = original_image.shape[:2]
        # Do not apply original augmentation, which is resize.
        # image = predictor.aug.get_transform(original_image).apply_image(original_image)
        image = original_image
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
        inputs = {"image": image, "height": height, "width": width}
        inputs_list.append(inputs)
        index = 0
        outputs_list = predictions = predictor.model(inputs_list)
        im = original_image
        v = Visualizer(
                    im[:, :, ::-1],
                    metadata=metadata,
                    scale=0.5,
                    instance_mode=ColorMode.IMAGE_BW
                    # remove the colors of unsegmented pixels. This option is only available for segmentation models
                )
        # out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        # # cv2_imshow(out.get_image()[:, :, ::-1])
        # cv2.imwrite(str(outdir / f"pred_{index}.jpg"), out.get_image()[:, :, ::-1])

        for outputs in outputs_list:
            # print(outputs)
            # print(outputs["instances"])
            out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        # cv2_imshow(out.get_image()[:, :, ::-1])
            cv2.imwrite(str(outdir / f"pred_{index}.jpg"), out.get_image()[:, :, ::-1])
