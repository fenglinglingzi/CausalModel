"""

本模块负责从标注 JSON 中解析视频目标轨迹与行为标签，
并将其转换为模型可用的特征矩阵与真值序列。

主要流程：
1. 解析 videorectangle → 目标轨迹（关键帧插值）
2. 解析 timelinelabels → 行为时间段
3. 将轨迹转换为 (cx, cy, w, h, conf) 特征
4. 输出：
   - features: np.ndarray, shape = (T, N * 5)
   - truths: List[str], 每一帧的行为标签
"""


import json, os
import requests
from typing import List, Dict

BASE_URL = "http://49.234.120.241:8080"
VIDEO_DIR = "./video"
TOKEN = ""

def download_video(video: str, id: int):
    headers = {"Authorization": TOKEN}
    url = BASE_URL + video

    try:
        print(f"[INFO] Downloading video (id={id}, video={video}) ...")
        r = requests.get(url, headers=headers)
        r.raise_for_status()
    except requests.exceptions.HTTPError as e:
        print(f"[ERROR] Download failed: {e}")
        return

    print(f"[SUCCESS] Download finished for id={id}")

    os.makedirs(VIDEO_DIR, exist_ok=True)
    path = os.path.join(VIDEO_DIR, f"{id}.mp4")
    with open(path, "wb") as f:
        f.write(r.content)

def interpolate_sequence(kf0: Dict, kf1: Dict):
    frames = {}

    if kf0["enabled"]:
        f0, f1 = kf0["frame"], kf1["frame"]
        x0, y0, w0, h0, r0 = kf0["x"], kf0["y"], kf0["width"], kf0["height"], kf0["rotation"]
        x1, y1, w1, h1, r1 = kf1["x"], kf1["y"], kf1["width"], kf1["height"], kf1["rotation"]
        for f in range(f0, f1):
            alpha = (f - f0) / (f1 - f0)

            frames[f] = (
                x0 + alpha * (x1 - x0),
                y0 + alpha * (y1 - y0),
                w0 + alpha * (w1 - w0),
                h0 + alpha * (h1 - h0),
                r0 + alpha * (r1 - r0),
            )

    else:
        frames[kf0["frame"]] = (
            kf0["x"], kf0["y"], kf0["width"], kf0["height"], kf0["rotation"]
        )

    return frames

def merge_objects(objects: List[Dict], total_frames: int):
    result = {t: [] for t in range(1, total_frames + 1)}
    for object in objects:
        label, frames = object["label"], object["frames"]
        for f, bbox in frames.items():
            if not bbox:
                continue
            result[f].append({"label": label, "bbox": bbox})
    return result

def merge_actions(actions: List[Dict], total_frames: int):
    result = {t: [] for t in range(1, total_frames + 1)}
    for action in actions:
        label, (start, end) = action["label"], action["range"]
        for f in range(start, end):
            result[f].append({"label": label})
    return result


def from_ls_json(file_path: str, download: bool = False) -> Dict[int, Dict]:
    """
    从 Label Studio 导出的 JSON 文件中解析视频标注结果。

    支持：
    - videorectangle ：目标轨迹
    - timelinelabels ：行为时间段

    参数
    ----
    file_path : str
        JSON 文件路径
    download : bool
        是否下载原视频文件

    返回
    ----
    Dict[int, Dict]
        以视频 ID（Label Studio 的 task id）为键的字典，每个视频包含：
        - "objects" : dict
            目标轨迹信息，结构为
            { frame_id: { "label": str, "bbox": (x, y, w, h) } }

        - "actions" : dict
            行为时间段信息，结构为
            { frame_id: { "label": str } }

    """

    with open(file_path, encoding="utf-8") as f:
        data = json.load(f)

    results = {}
    total_frames = 0

    for item in data:
        objects = []
        actions = []

        if download:
            download_video(video=item["data"]["video"], id=item["id"])

        for anno in item["annotations"]:
            for res in anno["result"]:
                value, type = res["value"], res["type"]

                # 物体识别框
                if type == "videorectangle":
                    label = value["labels"][0]  # no multiple labels
                    total_frames = value["framesCount"]
                    sequence = value["sequence"]

                    frames = {t: {} for t in range(1, total_frames + 1)}
                    for i in range(len(sequence)):
                        kf0 = sequence[i]
                        kf1 = sequence[i+1] if i < len(sequence) - 1 else sequence[i]

                        # LS 原始数据只有关键帧
                        # 需要通过插值逻辑补充过渡帧
                        for f, bbox in interpolate_sequence(kf0, kf1).items():
                            frames[f] = bbox

                    objects.append({"label": label, "frames": frames})

                # 时序标签
                if type == "timelinelabels":
                    label = value["timelinelabels"][0]  # no multiple labels  
                    rg = value["ranges"][0]             # no multiple ranges
                    start, end = rg["start"], rg["end"]

                    actions.append({"label": label, "range": (start, end)})

        results[item["id"]] = {
            "objects": merge_objects(objects, total_frames),
            "actions": merge_actions(actions, total_frames),
        }

    return results


def from_ls_json_min(file_path: str, download: bool = False) -> Dict[int, Dict]:
    """
    从 Label Studio 导出的 JSON 文件中解析视频标注结果。

    支持：
    - objects ：目标轨迹
    - actions ：行为时间段

    参数
    ----
    file_path : str
        JSON 文件路径
    download : bool
        是否下载原视频文件

    返回
    ----
    Dict[int, Dict]
        以视频 ID（Label Studio 的 task id）为键的字典，每个视频包含：
        - "objects" : dict
            目标轨迹信息，结构为
            { frame_id: { "label": str, "bbox": (x, y, w, h) } }

        - "actions" : dict
            行为时间段信息，结构为
            { frame_id: { "label": str } }

    """

    with open(file_path, encoding="utf-8") as f:
        data = json.load(f)

    results = {}
    total_frames = 0

    for item in data:
        objects = []
        actions = []

        if download:
            download_video(video=item["video"], id=item["id"])

        for object in item.get("objects", []):
            label = object["labels"][0]  # no multiple labels
            total_frames = object["framesCount"]
            sequence = object["sequence"]

            frames = {t: {} for t in range(1, total_frames + 1)}
            for i in range(len(sequence)):
                kf0 = sequence[i]
                kf1 = sequence[i+1] if i < len(sequence) - 1 else sequence[i]

                # LS 原始数据只有关键帧
                # 需要通过插值逻辑补充过渡帧
                for f, bbox in interpolate_sequence(kf0, kf1).items():
                    frames[f] = bbox

            objects.append({"label": label, "frames": frames})

        for action in item.get("actions", []):
            label = action["timelinelabels"][0]  # no multiple labels  
            rg = action["ranges"][0]             # no multiple ranges
            start, end = rg["start"], rg["end"]

            actions.append({"label": label, "range": (start, end)})

        results[item["id"]] = {
            "objects": merge_objects(objects, total_frames),
            "actions": merge_actions(actions, total_frames),
        }

    return results