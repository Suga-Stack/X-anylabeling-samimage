## 加载未适配的用户自定义模型

这里以分割模型 `segment` 为例，可遵循以下实施步骤：
（请重点关注 `a` `b` `e` ）

**a. 训练及导出模型**

导出 `ONNX` 模型，确保输出节点的维度为 `[1, C, H, W]`，其中 `C` 为总的类别数（包含背景类）。

**b. 定义配置文件**

首先，在[配置文件目录](../../anylabeling/configs/auto_labeling)下，新增一个配置文件，如`segment.yaml`：

```YAML
type: segment
name: segment1
display_name: segment1
provider: xxx
conf_threshold: 0.5
model_path: /path/to/best.onnx
classes:
  - cat
  - dog
  - _background_
```

其中：

| 字段 | 描述   |
|-----|--------|
| `type` | 指定模型类型，确保与现有模型类型不重复，以维护模型标识的唯一性。|
| `name` | 定义模型索引，用于内部引用和管理，避免与现有模型的索引名称冲突。|
| `display_name` | 展示在用户界面的模型名称，便于识别和选择，同样需保证其独特性，不与其它模型重名。|

以上三个字段为不可缺省字段。最后，可根据实际需要添加其它字段，如模型提供商、模型路径、模型超参等。

**c. 添加配置文件（封装端）**

其次，将上述配置文件添加到[模型管理文件](../../anylabeling/configs/models.yaml)中：

```yaml
...

- model_name: "segmengt1"
  config_file: ":/segment.yaml"
...

```

**d. 配置UI组件**

这一步可根据需要自行添加UI组件，只需将模型名称添加到对应的列表即可，具体可参考此[文件](../../anylabeling/services/auto_labeling/__init__.py) 中的定义。

**e. 定义推理服务(主要的模型代码)**

在定义推理服务的过程中，继承 [Model](../../anylabeling/services/auto_labeling/model.py) 基类是关键步骤之一，它允许你实现特定于模型的前向推理逻辑。
```py
import os
import pathlib
import yaml
import onnx
import urllib.request
import time
from urllib.parse import urlparse
from urllib.error import URLError

import ssl

ssl._create_default_https_context = (
    ssl._create_unverified_context
)  # Prevent issue when downloading models behind a proxy

import socket

socket.setdefaulttimeout(240)  # Prevent timeout when downloading models

from abc import abstractmethod

from PyQt5.QtCore import QCoreApplication, QFile, QObject
from PyQt5.QtGui import QImage

from .types import AutoLabelingResult
from anylabeling.config import get_config
from anylabeling.views.labeling.logger import logger
from anylabeling.views.labeling.label_file import LabelFile, LabelFileError


class Model(QObject):
    BASE_DOWNLOAD_URL = (
        "https://github.com/CVHub520/X-AnyLabeling/releases/tag"
    )

    # Add retry settings
    MAX_RETRIES = 2
    RETRY_DELAY = 3  # seconds

    class Meta(QObject):
        required_config_names = []
        widgets = ["button_run"]
        output_modes = {
            "rectangle": QCoreApplication.translate("Model", "Rectangle"),
        }
        default_output_mode = "rectangle"

    def __init__(self, model_config, on_message) -> None:
        super().__init__()
        self.on_message = on_message
        # Load and check config
        if isinstance(model_config, str):
            if not os.path.isfile(model_config):
                raise FileNotFoundError(
                    QCoreApplication.translate(
                        "Model", "Config file not found: {model_config}"
                    ).format(model_config=model_config)
                )
            with open(model_config, "r") as f:
                self.config = yaml.safe_load(f)
        elif isinstance(model_config, dict):
            self.config = model_config
        else:
            raise ValueError(
                QCoreApplication.translate(
                    "Model", "Unknown config type: {type}"
                ).format(type=type(model_config))
            )
        self.check_missing_config(
            config_names=self.Meta.required_config_names,
            config=self.config,
        )
        self.output_mode = self.Meta.default_output_mode
        self._config = get_config()

    def get_required_widgets(self):
        """
        Get required widgets for showing in UI
        """
        return self.Meta.widgets

    @staticmethod
    def allow_migrate_data():
        """Check if the current env have write permissions"""
        home_dir = os.path.expanduser("~")
        old_model_path = os.path.join(home_dir, "anylabeling_data")
        new_model_path = os.path.join(home_dir, "xanylabeling_data")

        if os.path.exists(new_model_path) or not os.path.exists(
            old_model_path
        ):
            return True

        if not os.access(home_dir, os.W_OK):
            return False

        try:
            os.rename(old_model_path, new_model_path)
            return True
        except Exception as e:
            logger.error(f"An error occurred during data migration: {str(e)}")
            return False

    def download_with_retry(self, url, dest_path, progress_callback):
        """Download file with retry mechanism"""
        for attempt in range(self.MAX_RETRIES):
            try:
                if attempt > 0:
                    logger.warning(
                        f"Retry attempt {attempt + 1}/{self.MAX_RETRIES}"
                    )
                urllib.request.urlretrieve(url, dest_path, progress_callback)
                return True
            except URLError as e:
                delay = self.RETRY_DELAY * (attempt + 1)
                if attempt < self.MAX_RETRIES - 1:
                    error_msg = f"Connection failed, retrying in {delay}s... (Attempt {attempt + 1}/{self.MAX_RETRIES} failed)"
                    logger.warning(error_msg)
                    self.on_message(error_msg)
                    time.sleep(delay)
                else:
                    logger.warning(
                        f"All download attempts failed ({self.MAX_RETRIES} tries)"
                    )
                    raise e

    def get_model_abs_path(self, model_config, model_path_field_name):
        """
        Get model absolute path from config path or download from url
        """
        # Try getting model path from config folder
        model_path = model_config[model_path_field_name]

        # Model path is a local path
        if not model_path.startswith(("http://", "https://")):
            # Relative path to executable or absolute path?
            model_abs_path = os.path.abspath(model_path)
            if os.path.exists(model_abs_path):
                return model_abs_path

            # Relative path to config file?
            config_file_path = model_config["config_file"]
            config_folder = os.path.dirname(config_file_path)
            model_abs_path = os.path.abspath(
                os.path.join(config_folder, model_path)
            )
            if os.path.exists(model_abs_path):
                return model_abs_path

            raise QCoreApplication.translate(
                "Model", "Model path not found: {model_path}"
            ).format(model_path=model_path)

        # Download model from url
        self.on_message(
            QCoreApplication.translate(
                "Model", "Downloading model from registry..."
            )
        )

        # Build download url
        def get_filename_from_url(url):
            a = urlparse(url)
            return os.path.basename(a.path)

        filename = get_filename_from_url(model_path)
        download_url = model_path

        # Continue with the rest of your function logic
        migrate_flag = self.allow_migrate_data()
        home_dir = os.path.expanduser("~")
        data_dir = "xanylabeling_data" if migrate_flag else "anylabeling_data"

        # Create model folder
        home_dir = os.path.expanduser("~")
        model_path = os.path.abspath(os.path.join(home_dir, data_dir))
        model_abs_path = os.path.abspath(
            os.path.join(
                model_path,
                "models",
                model_config["name"],
                filename,
            )
        )
        if os.path.exists(model_abs_path):
            if model_abs_path.lower().endswith(".onnx"):
                try:
                    onnx.checker.check_model(model_abs_path)
                except onnx.checker.ValidationError as e:
                    logger.error(f"{str(e)}")
                    logger.warning("Action: Delete and redownload...")
                    try:
                        os.remove(model_abs_path)
                        time.sleep(1)
                    except Exception as e:  # noqa
                        logger.error(f"Could not delete: {str(e)}")
                else:
                    return model_abs_path
            else:
                return model_abs_path
        pathlib.Path(model_abs_path).parent.mkdir(parents=True, exist_ok=True)

        # Download url
        use_modelscope = False
        env_model_hub = os.getenv("XANYLABELING_MODEL_HUB")
        if env_model_hub == "modelscope":
            use_modelscope = True
        elif (
            env_model_hub is None or env_model_hub == ""
        ):  # Only check config if env var is not set or empty
            if self._config.get("model_hub") == "modelscope":
                use_modelscope = True
            # Fallback to language check only if model_hub is not 'modelscope'
            elif (
                self._config.get("model_hub") is None
                or self._config.get("model_hub") == ""
            ):
                if self._config.get("language") == "zh_CN":
                    use_modelscope = True

        if use_modelscope:
            model_type = model_config["name"].split("-")[0]
            model_name = os.path.basename(download_url)
            download_url = f"https://www.modelscope.cn/models/CVHub520/{model_type}/resolve/master/{model_name}"

        ellipsis_download_url = download_url
        if len(download_url) > 40:
            ellipsis_download_url = (
                download_url[:20] + "..." + download_url[-20:]
            )

        logger.info(f"Downloading {download_url} to {model_abs_path}")
        try:

            def _progress(count, block_size, total_size):
                percent = int(count * block_size * 100 / total_size)
                self.on_message(
                    QCoreApplication.translate(
                        "Model", "Downloading {download_url}: {percent}%"
                    ).format(
                        download_url=ellipsis_download_url, percent=percent
                    )
                )

            self.download_with_retry(download_url, model_abs_path, _progress)

        except Exception as e:  # noqa
            logger.error(
                f"Could not download {download_url}: {e}, you can try to download it manually."
            )
            self.on_message(f"Download failed! Please try again later.")
            time.sleep(1)
            return None

        return model_abs_path

    def check_missing_config(self, config_names, config):
        """
        Check if config has all required config names
        """
        for name in config_names:
            if name not in config:
                raise Exception(f"Missing config: {name}")

    @abstractmethod
    def predict_shapes(self, image, filename=None) -> AutoLabelingResult:
        """
        Predict image and return AnyLabeling shapes
        """
        raise NotImplementedError

    @abstractmethod
    def unload(self):
        """
        Unload memory
        """
        raise NotImplementedError

    @staticmethod
    def load_image_from_filename(filename):
        """Load image from labeling file and return image data and image path."""
        label_file = os.path.splitext(filename)[0] + ".json"
        if QFile.exists(label_file) and LabelFile.is_label_file(label_file):
            try:
                label_file = LabelFile(label_file)
            except LabelFileError as e:
                logger.error("Error reading {}: {}".format(label_file, e))
                return None, None
            image_data = label_file.image_data
        else:
            image_data = LabelFile.load_image_file(filename)
        image = QImage.fromData(image_data)
        if image.isNull():
            logger.error("Error reading {}".format(filename))
        return image

    def on_next_files_changed(self, next_files):
        """
        Handle next files changed. This function can preload next files
        and run inference to save time for user.
        """
        pass

    def set_output_mode(self, mode):
        """
        Set output mode
        """
        self.output_mode = mode

```


具体地，你可以在[模型推理服务目录](../../anylabeling/services/auto_labeling/)下新建一个 `segment.py` 文件，参考示例如下：

```python
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


class Segment(Model):
    """Segment"""

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
        input_h, input_w = self.input_shape
        image = cv2.resize(input_image, (input_w, input_h))
        image = np.transpose(image, (2, 0, 1))
        image = image.astype(np.float32) / 255.0
        image = (image - 0.5) / 0.5
        image = np.expand_dims(image, axis=0)
        return image

    def postprocess(self, image, outputs):
        n, c, h, w = outputs.shape
        image_height, image_width = image.shape[:2]
        # Obtain the category index of each pixel
        # target shape: (1, h, w)
        outputs = np.argmax(outputs, axis=1)
        results = []
        for i in range(c):
            # Skip the background label
            if self.classes[i] == '_background_':
                continue
            # Get the category index of each pixel for the first batch by adding [0].
            mask = outputs[0] == i
            # Rescaled to original shape
            mask_resized = cv2.resize(mask.astype(np.uint8), (image_width, image_height))
            # Get the contours
            contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # Append the contours along with their respective class labels
            results.append((self.classes[i], [np.squeeze(contour).tolist() for contour in contours]))
        return results

    def predict_shapes(self, image, image_path=None):
        if image is None:
            return []

        try:
            image = qt_img_to_rgb_cv_img(image, image_path)
        except Exception as e:  # noqa
            logging.warning("Could not inference model")
            logging.warning(e)
            return []

        blob = self.preprocess(image)
        outputs = self.net.get_ort_inference(blob)
        results = self.postprocess(image, outputs)
        shapes = []
        for item in results:
            label, contours = item
            for points in contours:
                # Make sure to close
                points += points[0]
                shape = Shape(flags={})
                for point in points:
                    shape.add_point(QtCore.QPointF(point[0], point[1]))
                shape.shape_type = "polygon"
                shape.closed = True
                shape.fill_color = "#000000"
                shape.label = label
                shape.selected = False
                shapes.append(shape)

        result = AutoLabelingResult(shapes, replace=True)
        return result

    def unload(self):
        del self.net
```

这里：

- 元数据 `Meta` 类中：
    - `required_config_names`：用于指定模型配置文件中必须包含的配置项，确保模型推理服务能够正确初始化。
    - `widgets`：指定模型推理服务中需要显示的控件，如按钮、下拉框等，具体可参考此 [文件](../../anylabeling/services/auto_labeling/__init__.py) 中的定义。(可以保持默认)
    - `output_modes`：指定模型推理服务中输出的形状类型，支持多边形、矩形和旋转框等。
    - `default_output_mode`：指定模型推理服务中默认的输出形状类型。
- `predict_shapes` 和 `unload` 均属于抽象方法，分别用于定义模型推理过程和模型资源释放逻辑，因此一定需要实现。


**f. 添加至模型管理（封装端）**

完成上述步骤后，我们需要打开 [模型配置文件](../../anylabeling/services/auto_labeling/__init__.py) 中，并将对应的模型类型字段（如`segment`）添加至 `_CUSTOM_MODELS` 列表中，并根据需要在不同配置项中添加对应的模型名称。

> **提示**: 如果你不知道如何实现对应的控件，可打开搜索面板，输入相应关键字，查看所有可用控件的实现逻辑。

最后，移步至 [模型管理类文件](../../anylabeling/services/auto_labeling/model_manager.py) 中，在 `_load_model` 方法中按照如下方式初始化你的实例：


```python
...

class ModelManager(QObject):
    """Model manager"""

    def __init__(self):
        ...
    ...
    def _load_model(self, model_id):
        """Load and return model info"""
        if self.loaded_model_config is not None:
            self.loaded_model_config["model"].unload()
            self.loaded_model_config = None
            self.auto_segmentation_model_unselected.emit()

        model_config = copy.deepcopy(self.model_configs[model_id])
        if model_config["type"] == "yolov5":
            ...
        elif model_config["type"] == "segment":
            from .segment import Segment

            try:
                model_config["model"] = Segment(
                    model_config, on_message=self.new_model_status.emit
                )
                self.auto_segmentation_model_unselected.emit()
            except Exception as e:  # noqa
                self.new_model_status.emit(
                    self.tr(
                        "Error in loading model: {error_message}".format(
                            error_message=str(e)
                        )
                    )
                )
                print(
                    "Error in loading model: {error_message}".format(
                        error_message=str(e)
                    )
                )
                return
          ...
    ...
```

⚠️注意：

- 模型类型字段需要与上述步骤**b. 定义配置文件**中定义的配置文件中的 `type` 字段保持一致。
- 如果是基于 `SAM` 的模式，请将 `self.auto_segmentation_model_unselected.emit()` 替换为 `self.auto_segmentation_model_selected.emit()` 以触发相应的功能。