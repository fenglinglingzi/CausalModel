"""

本模块负责将 analyzer 输入转化为模型可用的特征矩阵。

1. 定义标准检测数据结构
   - Detection        : 单目标检测结果
   - FrameDetections  : 单帧中所有检测结果的集合

2. 提供 Mock 数据生成工具（用于测试和开发）
   - mock_detection          : 随机生成一个 Detection
   - mock_frame_detections    : 随机生成一帧检测结果
   - mock_windows             : 模拟 analyzer 的输入格式

3. 特征提取
   - window_to_features
     将多帧检测窗口转换为固定维度的特征矩阵，
     用于时序模型输入。

特征格式说明
- 每一类占用 5 个维度：
    [cx, cy, w, h, conf]
- 特征顺序按 class_id 排列：
    [
      cls0_cx, cls0_cy, cls0_w, cls0_h, cls0_conf,
      cls1_cx, cls1_cy, cls1_w, cls1_h, cls1_conf,
      ...
    ]
- 输出 shape：
    (num_frames, NUM_CLASSES * 5)

与标注系统（lab.py）输出的特征格式对齐

"""

import random
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


KNOWN_CLASSES = {
    0: "person",
    1: "car",
    2: "truck",
    3: "bicycle",
    4: "dog",
}

NUM_CLASSES = len(KNOWN_CLASSES)

WIDTH: int = 640
HEIGHT: int = 480


@dataclass
class Detection:
    """单个检测结果（标准格式）。所有检测模型的输出都应转换为此格式。"""

    bbox: List[int]  # [x1, y1, x2, y2]
    confidence: float  # 置信度 [0.0-1.0]
    class_id: int  # 类别 ID
    class_name: str  # 类别名称
    mask: Optional[np.ndarray] = None  # 分割掩码（可选）
    keypoints: Optional[List] = None  # 关键点（可选）
    extra: Dict[str, Any] = field(default_factory=dict)  # 扩展数据


@dataclass
class FrameDetections:
    """一帧里某检测器产出的全部检测（标准化输出，亦作推理最终输出格式）。"""

    detections: List[Detection]  # 检测结果列表
    metadata: Dict[str, Any]  # 元数据（如模型名称、推理时间等）
    timestamp: float  # 时间戳
    success: bool = True  # 推理是否成功
    error: Optional[str] = None  # 错误信息（失败时提供）


def mock_detection() -> Detection:
    """
    模拟 Detection 对象
    """
    w, h = WIDTH, HEIGHT

    x1 = random.randint(0, int(w * 0.8))
    y1 = random.randint(0, int(h * 0.8))
    x2 = random.randint(x1 + 10, w)
    y2 = random.randint(y1 + 10, h)

    class_id = random.randint(0, NUM_CLASSES - 1)

    return Detection(
        bbox=[x1, y1, x2, y2],
        confidence=round(random.uniform(0.3, 0.99), 2),
        class_id=class_id,
        class_name=KNOWN_CLASSES[class_id],
        mask=None,
        keypoints=None,
        extra={},
    )

def mock_frame_detections(
    num_detections: Optional[int] = None,
    timestamp: float = 0.0,
) -> FrameDetections:
    """
    模拟 FrameDetections 对象
    """
    if num_detections is None:
        num_detections = random.randint(0, 10)

    detections = [mock_detection() for _ in range(num_detections)]

    return FrameDetections(
        detections=detections,
        metadata={"model": "yolo"},
        timestamp=timestamp,
        success=True,
        error=None,
    )

def mock_windows(
    detectors=("yolo", "rtdetr"),
    num_frames=100,
    fps=30,
) -> Dict[str, List[FrameDetections]]:
    """
    模拟 analyze(self, windows: Dict[str, List[FrameDetections]]) 输入对象
    """
    windows = {}

    for detector in detectors:
        frames = []
        for i in range(num_frames):
            frames.append(mock_frame_detections(timestamp=i / fps))
        windows[detector] = frames

    return windows


def window_to_features(windows: Dict[str, List[FrameDetections]]) -> np.ndarray:
    """
    将 windows 转换为固定维度的特征矩阵
    输出 shape: (NUM_CLASSES * 5, num_frames)
    """

    detector_name = list(windows.keys())[0]
    frames = windows[detector_name]

    features = []

    for frame in frames:
        feat_vec = [0.0] * (NUM_CLASSES * 5)

        # 按 class_id 分组
        class_dets = {}
        for det in frame.detections:
            class_dets.setdefault(det.class_id, []).append(det)

        for class_id, dets in class_dets.items():
            if class_id >= NUM_CLASSES:
                continue

            # 取置信度最高的 detection
            best_det = max(dets, key=lambda d: d.confidence)
            x1, y1, x2, y2 = best_det.bbox
            conf = best_det.confidence

            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            w, h = x2 - x1, y2 - y1

            # backend 按 xyxy 格式读取 YOLO 结果
            # 所以这里需要归一化处理
            cx, cy = cx / WIDTH, cy / HEIGHT
            w, h = w / WIDTH, h / HEIGHT

            base = class_id * 5
            feat_vec[base:base+5] = [cx, cy, w, h, conf]

        features.append(feat_vec)

    return np.array(features, dtype=np.float32)


if __name__ == "__main__":
    features = window_to_features(mock_windows(num_frames=123)) 
    print(features.shape)
    print(features[0])