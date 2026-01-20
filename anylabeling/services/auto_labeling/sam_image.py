import warnings
warnings.filterwarnings("ignore")

import os
import cv2
import traceback
import numpy as np
import logging

from PyQt5 import QtCore
from PyQt5.QtCore import QCoreApplication

from anylabeling.app_info import __preferred_device__
from anylabeling.views.labeling.shape import Shape
from anylabeling.views.labeling.logger import logger
from anylabeling.views.labeling.utils.opencv import qt_img_to_rgb_cv_img

from .model import Model
from .types import AutoLabelingResult

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .samimage.segment_anything_training import sam_model_registry
    from .samimage.lora import Linear, MergedLinear, ConvLoRA, mark_only_lora_as_trainable, lora_state_dict
    from .samimage.segment_anything_training.modeling.transformer import Attention
    from .samimage.segment_anything_training.modeling.image_encoder import Attention as EncoderAttention
except ImportError as e:
    logger.warning(f"Import error: {e}")


class SAMImage(Model):
    """清华聚好看-SAM缺陷检测模型"""

    class Meta:
        """Meta class to define required configurations and UI elements."""

        required_config_names = [
            "type",
            "name",
            "display_name",
            "model_cfg",
            "model_path",
            "lora_r",
        ]
        widgets = [
            "output_label",
            "output_select_combobox",
            "button_add_rect",
            "button_clear",
            "toggle_preserve_existing_annotations",
            "mask_fineness_slider",
            "mask_fineness_value_label",
        ]
        output_modes = {
            "polygon": QCoreApplication.translate("Model", "多边形"),
            "rectangle": QCoreApplication.translate("Model", "矩形"),
            "mask": QCoreApplication.translate("Model", "掩码"),
        }
        default_output_mode = "polygon"  # 默认输出多边形格式，符合labelme要求

    def __init__(self, config_path, on_message) -> None:
        """Initialize the defect detection model with given configuration."""

        super().__init__(config_path, on_message)

        # 设备配置
        device_type = self.config.get("device_type", "cuda")
        if device_type == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
            logger.info("Using CUDA device")
        elif device_type == "mps" and torch.backends.mps.is_available():
            self.device = torch.device("mps")
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
            logger.info("Using MPS device")
        else:
            self.device = torch.device("cpu")
            logger.info("Using CPU device")

        # 模型配置
        self.model_cfg = self.config.get("model_cfg", "vit_b")
        self.model_path = self.get_model_abs_path(self.config, "model_path")
        self.lora_r = self.config.get("lora_r", 8)
        self.input_size = self.config.get("input_size", [1024, 1024])
        self.mask_threshold = self.config.get("mask_threshold", 0.5)
        
        # 最长边的目标尺寸（保持宽高比）
        self.max_input_size = 1024
        
        if not self.model_path or not os.path.isfile(self.model_path):
            logger.error(f"Model path not found: {self.model_path}")
            raise FileNotFoundError("SAM模型权重文件未找到")

        # 初始化模型
        self.net = None
        self._initialize_model()
        
        # 标注相关配置
        self.marks = []
        self.replace = True
        self.epsilon = 0.001
        
        # 类别配置
        self.classes = self.config.get("classes", ["defect"])
        
        # 缩放因子（用于坐标映射）
        self.scale_factor = 1.0
        self.original_size = None
        self.resized_size = None
        
        logger.info("清华聚好看-SAM缺陷检测模型初始化完成")

    def set_mask_fineness(self, epsilon: float):
        """设置轮廓平滑系数"""
        self.epsilon = max(0.001, min(0.1, float(epsilon)))

    def _calculate_resize_scale(self, image_h: int, image_w: int) -> tuple:
        """计算缩放因子，保持宽高比，最长边缩到max_input_size以内
        
        Args:
            image_h: 原图高度
            image_w: 原图宽度
            
        Returns:
            tuple: (scale_factor, resized_h, resized_w)
        """
        max_side = max(image_h, image_w)
        
        if max_side <= self.max_input_size:
            # 无需缩放
            return 1.0, image_h, image_w
        
        # 计算缩放因子
        scale = self.max_input_size / max_side
        resized_h = int(image_h * scale)
        resized_w = int(image_w * scale)
        
        return scale, resized_h, resized_w

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        """对输入图像进行缩放（保持宽高比，最长边缩到max_input_size以内）
        
        Args:
            image: 原始图像 (H, W, C)
            
        Returns:
            np.ndarray: 缩放后的图像
        """
        image_h, image_w = image.shape[:2]
        self.scale_factor, resized_h, resized_w = self._calculate_resize_scale(image_h, image_w)
        self.original_size = (image_h, image_w)
        self.resized_size = (resized_h, resized_w)
        
        if self.scale_factor == 1.0:
            return image
        
        # 使用cv2缩放
        resized_image = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
        logger.debug(f"图像缩放: {image_h}x{image_w} -> {resized_h}x{resized_w} (scale={self.scale_factor:.4f})")
        
        return resized_image

    def _scale_marks_to_resized(self) -> tuple:
        """将原始尺寸的marks缩放到缩放后的尺寸
        
        Returns:
            tuple: (point_coords, point_labels, box) - 缩放后的提示
        """
        point_coords, point_labels, box = None, None, None
        
        for m in self.marks:
            if not isinstance(m, dict) or 'type' not in m:
                continue
                
            if m['type'] == 'rectangle':
                data = m.get('data')
                if data is None:
                    continue
                    
                try:
                    if isinstance(data, (list, tuple, np.ndarray)):
                        arr = np.array(data, dtype=np.float32)
                        if arr.size == 4:
                            x0, y0, x1, y1 = arr.tolist()
                        elif arr.shape == (2, 2):
                            (x0, y0), (x1, y1) = arr.tolist()
                        else:
                            continue
                        
                        # 缩放坐标
                        box = [
                            float(max(0, x0 * self.scale_factor)),
                            float(max(0, y0 * self.scale_factor)), 
                            float(x1 * self.scale_factor),
                            float(y1 * self.scale_factor)
                        ]
                        logger.debug(f"矩形提示缩放: ({x0}, {y0}, {x1}, {y1}) -> ({box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f})")
                        break
                except Exception as e:
                    logger.warning(f"矩形提示缩放失败: {e}")
                    continue
                    
            elif m['type'] == 'point':
                logger.debug("检测到点提示")
                pass
        
        return point_coords, point_labels, box

    def _scale_results_back_to_original(self, results: list) -> list:
        """将缩放后的检测结果坐标映射回原始图像尺寸
        
        Args:
            results: 缩放图像上的检测结果
            
        Returns:
            list: 映射到原始尺寸的检测结果
        """
        if self.scale_factor == 1.0:
            return results
        
        scaled_results = []
        
        for label, contours in results:
            scaled_contours = []
            
            for points in contours:
                # 将每个点的坐标缩放回原始尺寸
                scaled_points = []
                for point in points:
                    # point 是 [x, y]
                    scaled_x = point[0] / self.scale_factor
                    scaled_y = point[1] / self.scale_factor
                    scaled_points.append([scaled_x, scaled_y])
                
                scaled_contours.append(scaled_points)
            
            scaled_results.append((label, scaled_contours))
        
        logger.debug(f"坐标已映射回原始尺寸 (scale={1/self.scale_factor:.4f})")
        return scaled_results


    def set_auto_labeling_marks(self, marks):
        """由 UI 注入交互标记（点/框）"""
        self.marks = marks or []

    def set_auto_labeling_preserve_existing_annotations_state(self, state: bool):
        """与 UI 的"保留已有标注"开关联动"""
        self.replace = not bool(state)

    def marks_to_prompts(self):
        """将 UI 的 marks 转换为 SAM 可用的提示（优先矩形框）
        
        返回:
            tuple: (point_coords, point_labels, box)
        """
        # 对于已经初始化过的缩放，使用缩放后的marks
        if self.scale_factor != 1.0:
            return self._scale_marks_to_resized()
        
        point_coords, point_labels, box = None, None, None
        
        for m in self.marks:
            if not isinstance(m, dict) or 'type' not in m:
                continue
                
            if m['type'] == 'rectangle':
                data = m.get('data')
                if data is None:
                    continue
                    
                # 兼容多种矩形数据格式
                try:
                    if isinstance(data, (list, tuple, np.ndarray)):
                        arr = np.array(data, dtype=np.float32)
                        if arr.size == 4:
                            x0, y0, x1, y1 = arr.tolist()
                        elif arr.shape == (2, 2):
                            (x0, y0), (x1, y1) = arr.tolist()
                        else:
                            continue
                            
                        # 确保坐标有效性并归一化
                        box = [
                            float(max(0, x0)),
                            float(max(0, y0)), 
                            float(x1),
                            float(y1)
                        ]
                        logger.debug(f"检测到矩形提示: {box}")
                        break
                except Exception as e:
                    logger.warning(f"矩形提示解析失败: {e}")
                    continue
                    
            elif m['type'] == 'point':
                # 支持点提示（扩展功能）
                logger.debug("检测到点提示")
                pass
                
        return point_coords, point_labels, box

    def _initialize_model(self):
        """初始化SAM模型，支持先加载 base 权重（例如 vit_b.pth），再加载小的 adapter/LoRA 权重。

        兼容以下情况：
        - 单一 model_path（历史行为，加载一个完整权重或部分权重）
        - 提供 base_model_path + model_path（推荐）：先加载 base，再加载 adapter（strict=False）
        """
        try:
            # 创建基础模型（不传 checkpoint）
            net = sam_model_registry[self.model_cfg](checkpoint=None)

            # 准备 LoRA 结构（替换相应层以便加载 adapter 权重）
            prepare_lora(self.model_cfg, net, self.lora_r)

            # --- 加载 base 权重 ---
            base_path = self.config.get("base_model_path") or self.model_path
            if not base_path or not os.path.isfile(base_path):
                logger.error(f"Base model path not found: {base_path}")
                raise FileNotFoundError(f"Base model not found: {base_path}")

            base_ckpt = torch.load(base_path, map_location="cpu")
            if isinstance(base_ckpt, dict) and "model_state_dict" in base_ckpt:
                base_state = base_ckpt["model_state_dict"]
            else:
                base_state = base_ckpt

            missing_keys, unexpected_keys = net.load_state_dict(base_state, strict=False)
            if missing_keys:
                logger.warning(f"Base load 缺失键: {missing_keys}")
            if unexpected_keys:
                logger.warning(f"Base load 意外键: {unexpected_keys}")

            # --- 如果有 adapter（小权重），再加载它以覆盖 LoRA 部分 ---
            adapter_path = self.config.get("adapter_path") or self.config.get("model_path")
            # 确保 adapter_path 存在且不是与 base 相同的文件
            if (
                adapter_path
                and os.path.isfile(adapter_path)
                and os.path.abspath(adapter_path) != os.path.abspath(base_path)
            ):
                adapter_ckpt = torch.load(adapter_path, map_location="cpu")
                if isinstance(adapter_ckpt, dict):
                    if "model_state_dict" in adapter_ckpt:
                        adapter_state = adapter_ckpt["model_state_dict"]
                    elif "state_dict" in adapter_ckpt:
                        adapter_state = adapter_ckpt["state_dict"]
                    else:
                        adapter_state = adapter_ckpt
                else:
                    adapter_state = adapter_ckpt

                missing_keys2, unexpected_keys2 = net.load_state_dict(adapter_state, strict=False)
                if missing_keys2:
                    logger.info(f"Adapter load - 缺失键: {missing_keys2}")
                if unexpected_keys2:
                    logger.info(f"Adapter load - 意外键: {unexpected_keys2}")
                logger.info(f"已加载 adapter 权重: {adapter_path}")

            # 最终设置
            net.mask_threshold = float(self.mask_threshold)
            self.net = net.to(self.device)
            self.net.eval()
            logger.info(f"模型加载成功: {self.model_cfg}, LoRA r={self.lora_r}")

        except Exception as e:
            logger.error(f"模型初始化失败: {e}")
            raise

    def preprocess(self, input_image):
        """预处理输入图像"""
        # 首先对图像进行缩放（保持宽高比，最长边缩到1024以内）
        input_image = self._resize_image(input_image)
        
        # 确保为3通道RGB
        if len(input_image.shape) < 3:
            input_image = input_image[:, :, np.newaxis]
        if input_image.shape[2] == 1:
            input_image = np.repeat(input_image, 3, axis=2)
        elif input_image.shape[2] == 4:
            input_image = input_image[:, :, :3]  # 移除alpha通道

        # 缩放后的图像尺寸
        h, w = input_image.shape[:2]

        # 转换为 3xHxW 的 float32 Tensor
        tensor_image = (
            torch.as_tensor(input_image.copy(), dtype=torch.float32, device=self.device)
            .permute(2, 0, 1)
            .contiguous()
        )

        # 将交互标记转换为 box 提示（使用缩放后的坐标）
        _, _, box = self.marks_to_prompts()
        box_tensor = None
        if box is not None:
            box_tensor = torch.as_tensor([box], dtype=torch.float32, device=self.device)

        dict_input = {
            'image': tensor_image,
            'boxes': box_tensor,
            'original_size': (h, w),
        }

        return [dict_input]

    def postprocess(self, image, outputs):
        """后处理模型输出，生成多边形轮廓"""
        if outputs is None or len(outputs) == 0:
            return []

        pred = outputs[0]
        masks = pred.get('masks', None)
        
        if masks is None:
            # 回退到低分辨率logits
            low_res = pred.get('low_res_logits', None)
            if low_res is None:
                return []
                
            # 阈值化并上采样
            thr = float(self.mask_threshold)
            masks = (low_res > thr).float()
            oh, ow = image.shape[:2]
            masks = F.interpolate(masks, size=(oh, ow), mode='bilinear', align_corners=False) > 0

        # 处理mask形状
        if masks.dim() == 3:
            masks = masks.unsqueeze(0)

        masks_np = masks[0].detach().to('cpu').numpy()  # C x H x W

        # 如果存在矩形提示，裁剪mask到矩形区域内
        _, _, maybe_box = self.marks_to_prompts()
        if maybe_box is not None:
            x0, y0, x1, y1 = [int(v) for v in maybe_box]
            H, W = masks_np.shape[-2:]
            x0, y0 = max(0, min(W - 1, x0)), max(0, min(H - 1, y0))
            x1, y1 = max(0, min(W, x1)), max(0, min(H, y1))
            if x1 > x0 and y1 > y0:
                box_mask = np.zeros((H, W), dtype=bool)
                box_mask[y0:y1, x0:x1] = True
                masks_np = masks_np & box_mask

        image_height, image_width = image.shape[:2]
        results = []
        label_name = self.classes[0] if self.classes else "defect"

        for ci in range(masks_np.shape[0]):
            mask_bool = masks_np[ci]  # H x W
            
            # 跳过空mask
            if not np.any(mask_bool):
                continue
                
            # 转换为uint8用于轮廓检测
            mask_u8 = (mask_bool.astype(np.uint8)) * 255

            # 形态学操作，去除小噪声
            kernel = np.ones((3, 3), np.uint8)
            mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)
            mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)

            # 提取轮廓
            contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            contour_points = []
            for cnt in contours:
                if cnt is None or len(cnt) < 3:
                    continue
                    
                # 多边形简化
                eps = self.epsilon * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, eps, True)
                pts = approx.reshape(-1, 2).tolist()
                
                # 过滤太小的多边形
                if len(pts) >= 3 and cv2.contourArea(approx) > 10:
                    contour_points.append(pts)

            if contour_points:
                results.append((label_name, contour_points))
                logger.debug(f"检测到 {len(contour_points)} 个缺陷区域")

        # 将坐标映射回原始图像尺寸
        results = self._scale_results_back_to_original(results)
        
        return results

    def predict_shapes(self, image, image_path=None):
        """主预测函数，返回AutoLabelingResult"""
        if image is None:
            return AutoLabelingResult([], replace=False)

        try:
            image = qt_img_to_rgb_cv_img(image, image_path)
        except Exception as e:
            logger.warning(f"图像转换失败: {e}")
            return AutoLabelingResult([], replace=False)

        # 重置缩放因子（为下一次预测准备）
        self.scale_factor = 1.0
        self.original_size = None
        self.resized_size = None

        # 先计算会使用的缩放因子（不实际缩放图像）
        image_h, image_w = image.shape[:2]
        max_side = max(image_h, image_w)
        if max_side > self.max_input_size:
            self.scale_factor = self.max_input_size / max_side
        
        # 检查是否有交互提示（此时scale_factor已正确设置）
        _, _, maybe_box = self.marks_to_prompts()
        if maybe_box is None:
            logger.debug("未检测到矩形提示，跳过预测")
            return AutoLabelingResult([], replace=False)

        try:
            # 预处理 -> 推理 -> 后处理
            blob = self.preprocess(image)
            with torch.no_grad():
                outputs_list, _ = self.net(blob, multimask_output=False)
            results = self.postprocess(image, outputs_list)
            
            # 转换为Shape对象
            shapes = self._create_shapes_from_results(results, image.shape)
            
            return AutoLabelingResult(shapes, replace=self.replace)
            
        except Exception as e:
            logger.error(f"预测过程出错: {e}")
            logger.error(traceback.format_exc())
            return AutoLabelingResult([], replace=False)

    def _create_shapes_from_results(self, results, image_shape):
        """将检测结果转换为Shape对象"""
        shapes = []
        
        for label, contours in results:
            for points in contours:
                if len(points) < 3:
                    continue
                    
                # 创建多边形形状
                shape = Shape(flags={})
                for point in points:
                    shape.add_point(QtCore.QPointF(point[0], point[1]))
                
                # 闭合多边形
                shape.add_point(QtCore.QPointF(points[0][0], points[0][1]))
                
                shape.shape_type = "polygon"
                shape.closed = True
                shape.fill_color = "#000000"
                shape.line_color = "#000000"
                shape.label = label
                shape.selected = False
                
                shapes.append(shape)
                
        return shapes

    def unload(self):
        """卸载模型，释放内存"""
        if hasattr(self, 'net') and self.net is not None:
            del self.net
            self.net = None
            
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        logger.info("模型已卸载")


def prepare_lora(model_type, model: nn.Module, r):
    """准备LoRA层"""
    for name, module in model.named_children():
        if 'neck' in name:
            continue
        if isinstance(module, Attention):
            # 替换Attention层的Q、V投影为LoRA版本
            q_proj = module.q_proj
            v_proj = module.v_proj
            new_q_proj = Linear(q_proj.in_features, q_proj.out_features, r=r)
            new_v_proj = Linear(v_proj.in_features, v_proj.out_features, r=r)
            setattr(module, 'q_proj', new_q_proj)
            setattr(module, 'v_proj', new_v_proj)
        elif isinstance(module, EncoderAttention):
            # 替换Encoder的QKV为LoRA版本
            qkv = module.qkv
            setattr(module, 'qkv', MergedLinear(qkv.in_features, qkv.out_features, r, enable_lora=[True, False, True]))
        elif ('rep' in model_type) and isinstance(module, nn.Conv2d) and module.kernel_size[0] == 1 and module.groups==1:
            # 替换1x1卷积为LoRA版本
            setattr(model, name, ConvLoRA(module, module.in_channels, module.out_channels, 1, r=r))
        else:
            prepare_lora(model_type, module, r)