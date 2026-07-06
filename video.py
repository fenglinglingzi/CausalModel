import cv2
import numpy as np
from typing import Dict


def plot_labeled_data(
    objects: Dict,
    actions: Dict,
    input_path: str,
    output_path: str,
    ls_fps: float = 24.0
):
    cap = cv2.VideoCapture(input_path)

    fps_real = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps_real, (W, H)
    )

    # 按 label 分配颜色
    all_labels = set()
    for frame_items in objects.values():
        for item in frame_items:
            all_labels.add(item["label"])

    color_map = {
        lid: tuple(np.random.randint(0, 256, 3).tolist())
        for lid in sorted(all_labels)
    }

    frame_id = 0  # OpenCV 0-based

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 对齐 LS 时间
        t_sec = frame_id / fps_real
        ls_frame = int(round(t_sec * ls_fps))

        # ---- 1. 画行为标签（顶部居中） ----
        if ls_frame in actions:
            labels = [a["label"] for a in actions[ls_frame]]
            action_text = " | ".join(labels)

            cv2.putText(
                frame, action_text,
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2, (0, 0, 255), 2, cv2.LINE_AA
            )

        # ---- 2. 画 bbox ----
        if ls_frame in objects:
            for obj in objects[ls_frame]:
                bbox = obj["bbox"]
                if not bbox:
                    continue

                x, y, w, h = bbox
                # 百分比 → 像素
                x1 = int(round(x / 100.0 * W))
                y1 = int(round(y / 100.0 * H))
                x2 = int(round((x + w) / 100.0 * W))
                y2 = int(round((y + h) / 100.0 * H))

                color = color_map[obj["label"]]

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, obj["label"], (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
                )

        out.write(frame)
        frame_id += 1

    cap.release()
    out.release()


if __name__ == "__main__":
    
    from lab import from_ls_json, from_ls_json_min

    id = 50
    file_path = "data/project-10-at-2026-07-06-02-05-4bc8a35f.json"
    # file_path = "data/project-10-at-2026-07-05-12-50-1e090dd7.json"

    data = from_ls_json(file_path, download=False)
    objects, actions = data[id]["objects"], data[id]["actions"]

    plot_labeled_data(
        objects=objects,
        actions=actions,
        input_path=f"video/{id}.mp4",
        output_path=f"{id}_labeled.mp4"
    )