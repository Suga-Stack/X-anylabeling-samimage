import logging
import os

import cv2
import numpy as np
from PyQt5 import QtCore
from PyQt5.QtCore import QCoreApplication

from anylabeling.app_info import __preferred_device__
from anylabeling.views.labeling.shape import Shape
from anylabeling.views.labeling.utils.opencv import qt_img_to_rgb_cv_img
from .model import Model
from .types import AutoLabelingResult
from .engines.build_onnx_engine import OnnxBaseModel


class DeepLabV3(Model):
    """Semantic segmentation model using DeepLabV3"""

    class Meta:
        required_config_names = [
            "type",
            "name",
            "display_name",
            "model_path",
            "classes",
        ]
        widgets = ["button_run"]
        output_modes = {
            "polygon": QCoreApplication.translate("Model", "Polygon"),
        }
        default_output_mode = "polygon"

    def __init__(self, model_config, on_message) -> None:
        # Run the parent class's init method
        super().__init__(model_config, on_message)
        model_name = self.config["type"]
        model_abs_path = self.get_model_abs_path(self.config, "model_path")
        if not model_abs_path or not os.path.isfile(model_abs_path):
            raise FileNotFoundError(
                QCoreApplication.translate(
                    "Model",
                    f"Could not download or initialize {model_name} model.",
                )
            )
        self.net = OnnxBaseModel(model_abs_path, __preferred_device__)
        self.classes = self.config["classes"]
        self.input_shape = self.net.get_input_shape()[-2:]

    def preprocess(self, input_image):
        """
        Pre-processes the input image before feeding it to the network.
        """
        # DeepLabV3 expects specific preprocessing
        input_h, input_w = self.input_shape
        image = cv2.resize(input_image, (input_w, input_h))
        
        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Apply ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (image - mean) / std
        
        # Change from HWC to CHW
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, axis=0)
        
        return image.astype(np.float32)

    def postprocess(self, image, outputs):
        """
        Post-processes the network's output.
        """
        # outputs shape: [1, num_classes, H, W]
        n, c, h, w = outputs.shape
        image_height, image_width = image.shape[:2]
        
        # Obtain the category index of each pixel
        # target shape: (1, h, w)
        outputs = np.argmax(outputs, axis=1)
        
        results = []
        for i in range(c):
            # Skip the background label (index 0)
            if i == 0:  # background class
                continue
                
            # Get the mask for current class
            mask = outputs[0] == i
            
            # Skip if no pixels of this class
            if not np.any(mask):
                continue
                
            # Resize to original image shape
            mask_resized = cv2.resize(
                mask.astype(np.uint8), 
                (image_width, image_height),
                interpolation=cv2.INTER_NEAREST
            )
            
            # Get the contours
            contours, _ = cv2.findContours(
                mask_resized, 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            # Filter small contours
            filtered_contours = []
            for contour in contours:
                if cv2.contourArea(contour) > 100:  # minimum area threshold
                    filtered_contours.append(np.squeeze(contour).tolist())
            
            if filtered_contours:
                results.append((self.classes[i], filtered_contours))
                
        return results

    def predict_shapes(self, image, image_path=None):
        """
        Predict shapes from image
        """
        if image is None:
            return []

        try:
            image = qt_img_to_rgb_cv_img(image, image_path)
        except Exception as e:
            logging.warning("Could not inference model")
            logging.warning(e)
            return []

        # Preprocess
        blob = self.preprocess(image)
        
        # Inference
        outputs = self.net.get_ort_inference(blob)
        
        # Postprocess
        results = self.postprocess(image, outputs)
        
        # Convert to shapes
        shapes = []
        for label, contours in results:
            for points in contours:
                if len(points) < 3:  # Need at least 3 points for a polygon
                    continue
                    
                # Make sure the polygon is closed
                if not (points[0] == points[-1]):
                    points.append(points[0])
                
                shape = Shape(flags={})
                for point in points:
                    if isinstance(point, list) and len(point) == 2:
                        shape.add_point(QtCore.QPointF(point[0], point[1]))
                    else:
                        # Handle single point format
                        shape.add_point(QtCore.QPointF(point, 0))
                        
                shape.shape_type = "polygon"
                shape.closed = True
                shape.fill_color = "#000000"
                shape.line_color = "#000000"
                shape.line_width = 1
                shape.label = label
                shape.selected = False
                shapes.append(shape)

        result = AutoLabelingResult(shapes, replace=True)
        return result

    def unload(self):
        del self.net