"""
This module provides functionality for YOLO object detection using ONNX models from Hugging Face.

It includes utilities for preprocessing images, performing object detection, and post-processing
the results. The main components are:

1. YOLOModel class: Manages YOLO models from a Hugging Face repository.
2. Helper functions for coordinate conversion, non-maximum suppression, and image processing.
3. A high-level function 'yolo_predict' for easy object detection on images.

The module supports various image input types and allows customization of confidence and IoU thresholds.
"""

import ast
import json
import math
import os
from functools import lru_cache
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image
from hfutils.utils import hf_fs_path, hf_normpath
from huggingface_hub import HfFileSystem, hf_hub_download

from ..data import load_image, rgb_encode, ImageTyping
from ..utils import open_onnx_model

__all__ = [
    'YOLOModel',
    'yolo_predict',
]


def _yolo_xywh2xyxy(x: np.ndarray) -> np.ndarray:
    """
    Convert bounding box coordinates from (x, y, width, height) format to (x1, y1, x2, y2) format.

    This function is adapted from YOLOv8 and transforms the center-based representation
    to a corner-based representation of bounding boxes.

    :param x: Input bounding box coordinates in (x, y, width, height) format.
    :type x: np.ndarray

    :return: Bounding box coordinates in (x1, y1, x2, y2) format.
    :rtype: np.ndarray

    :Example:

    >>> import numpy as np
    >>> boxes = np.array([[10, 10, 20, 20]])
    >>> _yolo_xywh2xyxy(boxes)
    array([[  0.,   0.,  20.,  20.]])
    """
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y


def _yolo_nms(boxes, scores, thresh: float = 0.7) -> List[int]:
    """
    Perform Non-Maximum Suppression (NMS) on bounding boxes.

    This function applies NMS to remove overlapping bounding boxes, keeping only the most confident detections.

    :param boxes: Array of bounding boxes, each in the format [xmin, ymin, xmax, ymax].
    :type boxes: np.ndarray
    :param scores: Array of confidence scores for each bounding box.
    :type scores: np.ndarray
    :param thresh: IoU threshold for considering boxes as overlapping. Default is 0.7.
    :type thresh: float

    :return: List of indices of the boxes to keep after NMS.
    :rtype: List[int]

    :Example:

    >>> boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [20, 20, 30, 30]])
    >>> scores = np.array([0.9, 0.8, 0.7])
    >>> _yolo_nms(boxes, scores, 0.5)
    [0, 2]
    """
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)

        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(iou <= thresh)[0]
        order = order[inds + 1]

    return keep


def _image_preprocess(image: Image.Image, max_infer_size: int = 1216, align: int = 32):
    """
    Preprocess an input image for inference.

    This function resizes the image while maintaining its aspect ratio, and ensures
    the dimensions are multiples of 'align'.

    :param image: Input image to be preprocessed.
    :type image: Image.Image
    :param max_infer_size: Maximum size (width or height) of the processed image. Default is 1216.
    :type max_infer_size: int
    :param align: Value to align the image dimensions to. Default is 32.
    :type align: int

    :return: A tuple containing:
        - The preprocessed image
        - Original image dimensions (width, height)
        - New image dimensions (width, height)
    :rtype: tuple(Image.Image, tuple(int, int), tuple(int, int))

    :Example:

    >>> from PIL import Image
    >>> img = Image.new('RGB', (1000, 800))
    >>> processed_img, old_size, new_size = _image_preprocess(img)
    >>> print(old_size, new_size)
    (1000, 800) (1216, 992)
    """
    old_width, old_height = image.width, image.height
    new_width, new_height = old_width, old_height
    r = max_infer_size / max(new_width, new_height)
    if r < 1:
        new_width, new_height = new_width * r, new_height * r
    new_width = int(math.ceil(new_width / align) * align)
    new_height = int(math.ceil(new_height / align) * align)
    image = image.resize((new_width, new_height))
    return image, (old_width, old_height), (new_width, new_height)


def _xy_postprocess(x, y, old_size, new_size):
    """
    Convert coordinates from the preprocessed image size back to the original image size.

    :param x: X-coordinate in the preprocessed image.
    :type x: float
    :param y: Y-coordinate in the preprocessed image.
    :type y: float
    :param old_size: Original image dimensions (width, height).
    :type old_size: tuple(int, int)
    :param new_size: Preprocessed image dimensions (width, height).
    :type new_size: tuple(int, int)

    :return: Adjusted (x, y) coordinates for the original image size.
    :rtype: tuple(int, int)

    :Example:

    >>> _xy_postprocess(100, 100, (1000, 800), (1216, 992))
    (82, 80)
    """
    old_width, old_height = old_size
    new_width, new_height = new_size
    x, y = x / new_width * old_width, y / new_height * old_height
    x = int(np.clip(x, a_min=0, a_max=old_width).round())
    y = int(np.clip(y, a_min=0, a_max=old_height).round())
    return x, y


def _data_postprocess(output, conf_threshold, iou_threshold, old_size, new_size, labels: List[str]):
    """
    Post-process the raw output from the object detection model.

    This function applies confidence thresholding, non-maximum suppression, and
    converts the coordinates back to the original image size.

    :param output: Raw output from the object detection model.
    :type output: np.ndarray
    :param conf_threshold: Confidence threshold for filtering detections.
    :type conf_threshold: float
    :param iou_threshold: IoU threshold for non-maximum suppression.
    :type iou_threshold: float
    :param old_size: Original image dimensions (width, height).
    :type old_size: tuple(int, int)
    :param new_size: Preprocessed image dimensions (width, height).
    :type new_size: tuple(int, int)
    :param labels: List of class labels.
    :type labels: List[str]

    :return: List of detections, each in the format ((x0, y0, x1, y1), label, confidence).
    :rtype: List[tuple(tuple(int, int, int, int), str, float)]

    :Example:

    >>> output = np.array([[10, 10, 20, 20, 0.9, 0.1]])
    >>> _data_postprocess(output, 0.5, 0.5, (100, 100), (128, 128), ['cat', 'dog'])
    [((7, 7, 15, 15), 'cat', 0.9)]
    """
    max_scores = output[4:, :].max(axis=0)
    output = output[:, max_scores > conf_threshold].transpose(1, 0)
    boxes = output[:, :4]
    scores = output[:, 4:]
    filtered_max_scores = scores.max(axis=1)

    if not boxes.size:
        return []

    boxes = _yolo_xywh2xyxy(boxes)
    idx = _yolo_nms(boxes, filtered_max_scores, thresh=iou_threshold)
    boxes, scores = boxes[idx], scores[idx]

    detections = []
    for box, score in zip(boxes, scores):
        x0, y0 = _xy_postprocess(box[0], box[1], old_size, new_size)
        x1, y1 = _xy_postprocess(box[2], box[3], old_size, new_size)
        max_score_id = score.argmax()
        detections.append(((x0, y0, x1, y1), labels[max_score_id], float(score[max_score_id])))

    return detections


def _safe_eval_names_str(names_str):
    """
    Safely evaluate the names string from model metadata.

    :param names_str: String representation of names dictionary.
    :type names_str: str
    :return: Dictionary of name mappings.
    :rtype: dict
    :raises RuntimeError: If an invalid key or value type is encountered.

    This function parses the names string from the model metadata, ensuring that
    only string and number literals are evaluated for safety.
    """
    node = ast.parse(names_str, mode='eval')
    result = {}
    # noinspection PyUnresolvedReferences
    for key, value in zip(node.body.keys, node.body.values):
        if isinstance(key, (ast.Str, ast.Num)):
            key = ast.literal_eval(key)
        else:
            raise RuntimeError(f"Invalid key type: {key!r}, this should be a bug, "
                               f"please open an issue to dghs-imgutils.")  # pragma: no cover

        if isinstance(value, (ast.Str, ast.Num)):
            value = ast.literal_eval(value)
        else:
            raise RuntimeError(f"Invalid value type: {value!r}, this should be a bug, "
                               f"please open an issue to dghs-imgutils.")  # pragma: no cover

        result[key] = value

    return result


class YOLOModel:
    """
    A class to manage YOLO models from a Hugging Face repository.

    This class handles the loading, caching, and inference of YOLO models.

    :param repo_id: The Hugging Face repository ID containing the YOLO models.
    :type repo_id: str
    :param hf_token: Optional Hugging Face authentication token.
    :type hf_token: Optional[str]

    :Example:

    >>> model = YOLOModel("username/repo_name")
    >>> image = Image.open("path/to/image.jpg")
    >>> detections = model.predict(image, "model_name")
    """

    def __init__(self, repo_id: str, hf_token: Optional[str] = None):
        """
        Initialize the YOLOModel.

        :param repo_id: The Hugging Face repository ID containing the YOLO models.
        :type repo_id: str
        :param hf_token: Optional Hugging Face authentication token.
        :type hf_token: Optional[str]
        """
        self.repo_id = repo_id
        self._model_names = None
        self._models = {}
        self._hf_token = hf_token

    def _get_hf_token(self) -> Optional[str]:
        """
        Get the Hugging Face token, either from the instance or environment variable.

        :return: Hugging Face token.
        :rtype: Optional[str]
        """
        return self._hf_token or os.environ.get('HF_TOKEN')

    @property
    def model_names(self) -> List[str]:
        """
        Get the list of available model names in the repository.

        :return: List of model names.
        :rtype: List[str]
        """
        if self._model_names is None:
            hf_fs = HfFileSystem(token=self._get_hf_token())
            self._model_names = [
                hf_normpath(os.path.dirname(os.path.relpath(item, self.repo_id)))
                for item in hf_fs.glob(hf_fs_path(
                    repo_id=self.repo_id,
                    repo_type='model',
                    filename='*/model.onnx',
                ))
            ]

        return self._model_names

    def _check_model_name(self, model_name: str):
        """
        Check if the given model name is valid for this repository.

        :param model_name: Name of the model to check.
        :type model_name: str
        :raises ValueError: If the model name is not found in the repository.
        """
        if model_name not in self.model_names:
            raise ValueError(f'Unknown model {model_name!r} in model repository {self.repo_id!r}, '
                             f'models {self.model_names!r} are available.')

    def _open_model(self, model_name: str):
        """
        Open and cache a YOLO model.

        :param model_name: Name of the model to open.
        :type model_name: str
        :return: Tuple containing the ONNX model, maximum inference size, and labels.
        :rtype: tuple
        """
        if model_name not in self._models:
            self._check_model_name(model_name)
            model = open_onnx_model(hf_hub_download(
                self.repo_id,
                f'{model_name}/model.onnx',
                token=self._get_hf_token(),
            ))
            model_metadata = model.get_modelmeta()
            if 'imgsz' in model_metadata.custom_metadata_map:
                max_infer_size = max(json.loads(model_metadata.custom_metadata_map['imgsz']))
            else:
                max_infer_size = 640
            names_map = _safe_eval_names_str(model_metadata.custom_metadata_map['names'])
            labels = [names_map[i] for i in range(len(names_map))]
            self._models[model_name] = (model, max_infer_size, labels)

        return self._models[model_name]

    def predict(self, image: ImageTyping, model_name: str,
                conf_threshold: float = 0.25, iou_threshold: float = 0.7) \
            -> List[Tuple[Tuple[int, int, int, int], str, float]]:
        """
        Perform object detection on an image using the specified YOLO model.

        :param image: Input image for object detection.
        :type image: ImageTyping
        :param model_name: Name of the YOLO model to use.
        :type model_name: str
        :param conf_threshold: Confidence threshold for filtering detections. Default is 0.25.
        :type conf_threshold: float
        :param iou_threshold: IoU threshold for non-maximum suppression. Default is 0.7.
        :type iou_threshold: float

        :return: List of detections, each in the format ((x0, y0, x1, y1), label, confidence).
        :rtype: List[Tuple[Tuple[int, int, int, int], str, float]]

        :Example:

        >>> model = YOLOModel("username/repo_name")
        >>> image = Image.open("path/to/image.jpg")
        >>> detections = model.predict(image, "model_name")
        >>> print(detections[0])  # First detection
        ((100, 200, 300, 400), 'person', 0.95)
        """
        model, max_infer_size, labels = self._open_model(model_name)
        image = load_image(image, mode='RGB')
        new_image, old_size, new_size = _image_preprocess(image, max_infer_size)
        data = rgb_encode(new_image)[None, ...]
        output, = model.run(['output0'], {'images': data})
        return _data_postprocess(output[0], conf_threshold, iou_threshold, old_size, new_size, labels)

    def clear(self):
        """
        Clear cached model and metadata.
        """
        self._models.clear()


@lru_cache()
def _open_models_for_repo_id(repo_id: str, hf_token: Optional[str] = None) -> YOLOModel:
    """
    Load and cache a YOLO model from a Hugging Face repository.

    This function uses the `lru_cache` decorator to cache the loaded models,
    improving performance for repeated calls with the same repository ID.

    :param repo_id: The Hugging Face repository ID for the YOLO model.
    :type repo_id: str
    :param hf_token: Optional Hugging Face authentication token.
    :type hf_token: Optional[str]

    :return: The loaded YOLO model.
    :rtype: YOLOModel

    :raises Exception: If there's an error loading the model from the repository.

    Usage:
        >>> model = _open_models_for_repo_id("yolov5/yolov5s")
        >>> # Subsequent calls with the same repo_id will return the cached model
        >>> same_model = _open_models_for_repo_id("yolov5/yolov5s")
    """
    return YOLOModel(repo_id, hf_token=hf_token)


def yolo_predict(image: ImageTyping, repo_id: str, model_name: str,
                 conf_threshold: float = 0.25, iou_threshold: float = 0.7,
                 hf_token: Optional[str] = None) \
        -> List[Tuple[Tuple[int, int, int, int], str, float]]:
    """
    Perform object detection on an image using a YOLO model from a Hugging Face repository.

    This function is a high-level wrapper around the YOLOModel class, providing a simple
    interface for object detection without needing to explicitly manage model instances.

    :param image: Input image for object detection.
    :type image: ImageTyping
    :param repo_id: The Hugging Face repository ID containing the YOLO models.
    :type repo_id: str
    :param model_name: Name of the YOLO model to use.
    :type model_name: str
    :param conf_threshold: Confidence threshold for filtering detections. Default is 0.25.
    :type conf_threshold: float
    :param iou_threshold: IoU threshold for non-maximum suppression. Default is 0.7.
    :type iou_threshold: float
    :param hf_token: Optional Hugging Face authentication token.
    :type hf_token: Optional[str]

    :return: List of detections, each in the format ((x0, y0, x1, y1), label, confidence).
    :rtype: List[Tuple[Tuple[int, int, int, int], str, float]]

    :Example:

    >>> from PIL import Image
    >>> image = Image.open("path/to/image.jpg")
    >>> detections = yolo_predict(image, "username/repo_name", "model_name")
    >>> print(detections[0])  # First detection
    ((100, 200, 300, 400), 'person', 0.95)
    """
    return _open_models_for_repo_id(repo_id, hf_token=hf_token).predict(
        image=image,
        model_name=model_name,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
    )
