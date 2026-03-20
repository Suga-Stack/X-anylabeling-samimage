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
        default_output_mode = "polygon"

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
        self.marks = []               # 原始marks（用于调试）
        self.prompt_marks = []        # 用于SAM提示的非背景标记
        self.exclusion_marks = []      # 用于背景排除的标记
        self.replace = True
        self.epsilon = 0.001
        
        # 类别配置
        self.classes = self.config.get("classes", ["defect"])
        
        # 缩放因子（用于坐标映射）
        self.scale_factor = 1.0
        self.original_size = None
        self.resized_size = None
        
        # 调试图片保存路径
        self.debug_dir = "sam_debug"
        if not os.path.exists(self.debug_dir):
            os.makedirs(self.debug_dir)
            
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

    def _scale_marks_to_resized(self, marks_list) -> tuple:
        """将原始尺寸的marks缩放到缩放后的尺寸，只考虑非背景标记
        
        Args:
            marks_list: 要缩放的marks列表（通常为 self.prompt_marks）
        
        Returns:
            tuple: (point_coords, point_labels, box) - 缩放后的提示
        """
        point_coords, point_labels, box = None, None, None
        
        logger.debug(f"_scale_marks_to_resized: 处理 {len(marks_list)} 个marks")
        for idx, m in enumerate(marks_list):
            logger.debug(f"  检查mark {idx}: {m}")
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
                        logger.info(f"  选中非背景矩形提示: ({x0}, {y0}, {x1}, {y1}) -> 缩放后 {box}")
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
        """由 UI 注入交互标记（点/框），分离背景标记和提示标记"""
        self.marks = marks or []
        # 分类
        self.prompt_marks = []
        self.exclusion_marks = []
        for m in self.marks:
            if not isinstance(m, dict):
                continue
            label = m.get('label') or m.get('name') or ''
            is_bg = str(label).lower() in ('_background_', 'background', 'bg')
            if is_bg or m.get('is_exclusion') or m.get('exclude'):
                self.exclusion_marks.append(m)
            else:
                self.prompt_marks.append(m)
        logger.info(f"set_auto_labeling_marks: prompt={len(self.prompt_marks)}, exclusion={len(self.exclusion_marks)}")

    def set_auto_labeling_preserve_existing_annotations_state(self, state: bool):
        """与 UI 的"保留已有标注"开关联动"""
        self.replace = not bool(state)

    def marks_to_prompts(self):
        """将 self.prompt_marks 转换为 SAM 可用的提示（优先矩形框）
        
        返回:
            tuple: (point_coords, point_labels, box) - 仅包含非背景的提示
        """
        # 对于已经初始化过的缩放，使用缩放后的marks
        if self.scale_factor != 1.0:
            return self._scale_marks_to_resized(self.prompt_marks)
        
        point_coords, point_labels, box = None, None, None
        
        logger.debug(f"marks_to_prompts: 共有 {len(self.prompt_marks)} 个提示标记")
        for idx, m in enumerate(self.prompt_marks):
            logger.debug(f"  检查提示标记 {idx}: {m}")
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
                        logger.info(f"  选中非背景矩形提示: {box}")
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
        """初始化SAM模型，支持先加载 base 权重，再加载 adapter/LoRA 权重。"""
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

    def _has_exclusion_marks(self):
        """检查是否有真正的排除标记（严格判断）"""
        return len(self.exclusion_marks) > 0

    def _build_exclusion_mask(self, h: int, w: int, target_size: tuple = None):
        """根据 self.exclusion_marks 构建一个布尔型的排除掩码。
        
        只处理明确标记为背景的形状。
        
        Args:
            h: 原始高度（用于日志）
            w: 原始宽度（用于日志）
            target_size: 目标尺寸 (height, width)，如果提供，会将排除掩码缩放到此尺寸
        
        返回: np.ndarray(dtype=bool) 或 None
        """
        if not self.exclusion_marks:
            logger.debug("exclusion_marks为空，跳过排除掩码构建")
            return None

        # 在原始图像尺寸上构建掩码
        original_h, original_w = self.original_size if self.original_size else (h, w)
        # 使用 uint8 类型构建掩码，因为 OpenCV 函数需要这个类型
        mask_uint8 = np.zeros((original_h, original_w), dtype=np.uint8)
        bg_marks_count = 0
        
        for idx, m in enumerate(self.exclusion_marks):
            try:
                if not isinstance(m, dict) or 'type' not in m:
                    continue

                # 记录找到的背景标记
                bg_marks_count += 1
                logger.debug(f"处理背景标记 {idx}: type={m['type']}, label={m.get('label')}")
                
                # 处理多边形/涂鸦
                if m['type'] in ['polygon', 'scribble', 'freeform']:
                    data = m.get('data')
                    if data is None or len(data) < 3:
                        continue
                        
                    try:
                        pts = np.array(data, dtype=np.int32)
                        if pts.size >= 6:  # 至少3个点
                            # 确保掩码是 uint8 类型
                            cv2.fillPoly(mask_uint8, [pts.reshape(-1, 2)], 255)
                            logger.debug(f"  填充多边形，点数: {len(pts)}")
                    except Exception as e:
                        logger.warning(f"多边形处理失败: {e}")
                        # 打印更多调试信息
                        logger.warning(f"  数据类型: mask_uint8.dtype={mask_uint8.dtype}, pts.shape={pts.shape}")
                        continue
                
                # 处理矩形
                elif m['type'] == 'rectangle':
                    data = m.get('data')
                    if data is None:
                        continue
                        
                    try:
                        # 解析矩形坐标
                        if isinstance(data, (list, tuple, np.ndarray)):
                            arr = np.array(data, dtype=np.float32).flatten()
                            if len(arr) >= 4:
                                x0, y0 = float(arr[0]), float(arr[1])
                                x1, y1 = float(arr[2]), float(arr[3])
                                
                                logger.debug(f"  背景矩形原始坐标: ({x0:.1f}, {y0:.1f}) -> ({x1:.1f}, {y1:.1f})")
                                
                                # 确保坐标有序
                                x0, x1 = min(x0, x1), max(x0, x1)
                                y0, y1 = min(y0, y1), max(y0, y1)
                                
                                # 转换为整数并裁剪到图像范围内
                                x0i = int(max(0, min(original_w-1, x0)))
                                y0i = int(max(0, min(original_h-1, y0)))
                                x1i = int(max(0, min(original_w, x1)))
                                y1i = int(max(0, min(original_h, y1)))
                                
                                if x1i > x0i and y1i > y0i:
                                    mask_uint8[y0i:y1i, x0i:x1i] = 255
                                    logger.debug(f"  填充原始矩形: [{x0i},{y0i},{x1i},{x1i-x0i} x {y1i-y0i}]")
                                else:
                                    logger.debug(f"  背景矩形无效: [{x0i},{y0i},{x1i},{y1i}]")
                    except Exception as e:
                        logger.warning(f"矩形处理失败: {e}")
                        continue
                
                # 处理mask
                elif m['type'] == 'mask':
                    data = m.get('data')
                    if data is None:
                        continue
                        
                    try:
                        arr = np.array(data)
                        if arr.shape[:2] == (original_h, original_w):
                            mask_uint8[arr > 0] = 255
                            logger.debug(f"  直接使用mask，形状: {arr.shape}")
                        else:
                            # 缩放mask到原始尺寸
                            arr_u8 = (arr.astype(np.uint8) * 255)
                            arr_rs = cv2.resize(arr_u8, (original_w, original_h), interpolation=cv2.INTER_NEAREST)
                            mask_uint8[arr_rs > 0] = 255
                            logger.debug(f"  缩放mask: {arr.shape} -> ({original_h},{original_w})")
                    except Exception as e:
                        logger.warning(f"mask处理失败: {e}")
                        continue

            except Exception as e:
                logger.warning(f"处理背景标记时出错: {e}")
                continue

        # 转换为布尔掩码
        mask = mask_uint8 > 0

        # 如果需要缩放到目标尺寸
        if target_size is not None and (target_size[0] != original_h or target_size[1] != original_w):
            target_h, target_w = target_size
            logger.debug(f"缩放排除掩码: {original_h}x{original_w} -> {target_h}x{target_w}")
            
            # 缩放掩码（使用 uint8 版本进行缩放，避免类型问题）
            mask_resized = cv2.resize(mask_uint8, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
            mask = mask_resized > 0
            
            # 打印缩放后的非零像素数
            logger.debug(f"缩放后排除掩码非零像素: {np.count_nonzero(mask)}/{target_h*target_w}")

        nnz = int(np.count_nonzero(mask))
        logger.info(f"[SAMImage] 背景标记统计: 找到 {bg_marks_count} 个背景标记, 排除掩码非零像素: {nnz}/{mask.size} ({nnz/mask.size*100:.2f}%)")

        if nnz == 0:
            logger.debug("排除掩码为空，返回None")
            return None

        return mask

    def postprocess(self, image, outputs):
        """后处理模型输出，生成多边形轮廓"""
        if outputs is None or len(outputs) == 0:
            return []

        pred = outputs[0]
        masks = pred.get('masks', None)
        
        if masks is None:
            low_res = pred.get('low_res_logits', None)
            if low_res is None:
                return []
            thr = float(self.mask_threshold)
            masks = (low_res > thr).float()
            oh, ow = image.shape[:2]
            masks = F.interpolate(masks, size=(oh, ow), mode='bilinear', align_corners=False) > 0

        if masks.dim() == 3:
            masks = masks.unsqueeze(0)

        masks_np = masks[0].detach().to('cpu').numpy()

        # 应用矩形框裁剪（仅对非背景提示框）
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

        # 获取掩码的实际尺寸
        mask_h, mask_w = masks_np.shape[-2:]
        results = []
        label_name = self.classes[0] if self.classes else "defect"

        # 只有在有真正背景标记时才构建排除掩码
        exclusion_mask = None
        if self._has_exclusion_marks():
            # 传入掩码尺寸作为目标尺寸
            exclusion_mask = self._build_exclusion_mask(
                mask_h, 
                mask_w, 
                target_size=(mask_h, mask_w)
            )
            if exclusion_mask is not None:
                logger.info(f"应用排除掩码，非零像素: {np.count_nonzero(exclusion_mask)}/{mask_h*mask_w}")

        # 保存调试图片的计数器
        debug_saved = False

        for ci in range(masks_np.shape[0]):
            mask_bool = masks_np[ci].copy()  # 使用copy避免修改原数组
            
            # 记录原始面积
            original_area = np.count_nonzero(mask_bool)
            logger.debug(f"mask {ci}: 原始面积 = {original_area}")
            
            # 应用排除掩码（如果有）
            if exclusion_mask is not None:
                try:
                    # 计算重叠区域
                    overlap = np.count_nonzero(mask_bool & exclusion_mask)
                    logger.debug(f"mask {ci}: 与排除掩码重叠 = {overlap}")
                    
                    # 确保尺寸匹配
                    if mask_bool.shape == exclusion_mask.shape:
                        # 保存调试图片（仅保存第一次）
                        if not debug_saved:
                            cv2.imwrite(os.path.join(self.debug_dir, 'original_mask.png'), (mask_bool.astype(np.uint8)*255))
                            cv2.imwrite(os.path.join(self.debug_dir, 'exclusion_mask.png'), (exclusion_mask.astype(np.uint8)*255))
                            
                        # 使用逻辑与操作排除背景区域
                        mask_bool = mask_bool & (~exclusion_mask)
                        
                        if not debug_saved:
                            cv2.imwrite(os.path.join(self.debug_dir, 'result_mask.png'), (mask_bool.astype(np.uint8)*255))
                            debug_saved = True
                            logger.info(f"调试图片已保存到 {self.debug_dir} 目录")
                        
                        after_nz = np.count_nonzero(mask_bool)
                        removed = original_area - after_nz
                        logger.debug(f"mask {ci}: 排除后面积 = {after_nz}, 排除像素 = {removed}")
                        
                        if removed > 0 and after_nz == 0:
                            logger.debug(f"  mask {ci} 被完全排除")
                    else:
                        logger.warning(f"尺寸不匹配: mask {mask_bool.shape}, exclusion {exclusion_mask.shape}")
                        
                except Exception as e:
                    logger.error(f"应用排除掩码时出错: {e}")
                    continue
            
            if not np.any(mask_bool):
                logger.debug(f"mask {ci}: 排除后为空，跳过")
                continue
                
            mask_u8 = (mask_bool.astype(np.uint8)) * 255
            kernel = np.ones((3, 3), np.uint8)
            mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)
            mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            contour_points = []
            for cnt in contours:
                if cnt is None or len(cnt) < 3:
                    continue
                eps = self.epsilon * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, eps, True)
                pts = approx.reshape(-1, 2).tolist()
                if len(pts) >= 3 and cv2.contourArea(approx) > 10:
                    contour_points.append(pts)

            if contour_points:
                results.append((label_name, contour_points))
                logger.debug(f"mask {ci}: 生成 {len(contour_points)} 个轮廓")
            else:
                logger.debug(f"mask {ci}: 没有有效轮廓")

        # 将坐标映射回原始图像尺寸
        results = self._scale_results_back_to_original(results)
        
        logger.info(f"后处理完成，找到 {len(results)} 个缺陷区域")
        return results

    def predict_shapes(self, image, image_path=None):
        """主预测函数，返回AutoLabelingResult"""
        if image is None:
            return AutoLabelingResult([], replace=False)

        # 调试：打印所有marks
        logger.info("="*50)
        logger.info(f"开始预测，marks总数: {len(self.marks)}")
        for i, m in enumerate(self.marks or []):
            logger.info(f"  mark {i}: type={m.get('type')}, label={m.get('label')}, is_exclusion={m.get('is_exclusion')}, data={m.get('data')}")
        logger.info(f"提示标记数: {len(self.prompt_marks)}, 背景标记数: {len(self.exclusion_marks)}")
        logger.info("="*50)

        try:
            image = qt_img_to_rgb_cv_img(image, image_path)
        except Exception as e:
            logger.warning(f"图像转换失败: {e}")
            return AutoLabelingResult([], replace=False)

        # 记录原始尺寸
        self.original_size = (image.shape[0], image.shape[1])
        
        # 重置缩放因子（为下一次预测准备）
        self.scale_factor = 1.0
        self.resized_size = None

        # 先计算会使用的缩放因子（不实际缩放图像）
        image_h, image_w = image.shape[:2]
        max_side = max(image_h, image_w)
        if max_side > self.max_input_size:
            self.scale_factor = self.max_input_size / max_side
        
        # 检查是否有非背景矩形提示
        _, _, maybe_box = self.marks_to_prompts()
        if maybe_box is None:
            logger.warning("未检测到非背景矩形提示，跳过预测")
            return AutoLabelingResult([], replace=False)
        else:
            logger.info(f"检测到非背景矩形提示框: {maybe_box}")

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