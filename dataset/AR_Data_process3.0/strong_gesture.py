#!/usr/bin/env python3
import os
import numpy as np
import torch
from transformers import CLIPImageProcessor, CLIPVisionModel
import cv2
from PIL import Image
import mediapipe as mp
import json
from datetime import datetime, timedelta, timezone

# ==================== 配置 ====================
BASE_DIR = r"D:\AR_dataset\dataset"
# get_timestamp.py 生成的文件在这里
INPUT_DATA_DIR = r"E:\AR_data_process3.0\data" 
# 结果保存目录
OUTPUT_DIR = r"E:\AR_data_process3.0\data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CLIP_MODEL_PATH = r"E:\AR_data_process3.0\models\clir教师模型"

# 映射关系 (保持你的配置)
AVI_TO_MP4_MAP = {
    # =========================== office ==============================
    # "Video_20260306_152340690.avi": "interaction_20260306_072344.mp4",  # 菜单      # Bian
    # "Video_20260227_202553335.avi": "interaction_20260227_122606.mp4",  # 选择
    # "Video_20260227_202953348.avi": "interaction_20260227_122952.mp4",  # 放大
    # "Video_20260227_203348219.avi": "interaction_20260227_123354.mp4",  # 缩小
    # "Video_20260227_204553897.avi": "interaction_20260227_124559.mp4",  # 画笔
    # "Video_20260227_203753817.avi": "interaction_20260227_123745.mp4",  # 取消

    "Video_20260131_200029359.avi": "interaction_20260131_120024.mp4",  # 菜单      # Luo
    "Video_20260227_213001434.avi": "interaction_20260227_132951.mp4",  # 选择
    "Video_20260227_213404452.avi": "interaction_20260227_133408.mp4",  # 放大
    "Video_20260131_194205407.avi": "interaction_20260131_114156.mp4",  # 缩小
    "Video_20260131_195202906.avi": "interaction_20260131_115150.mp4",  # 画笔
    "Video_20260131_194854095.avi": "interaction_20260131_114852.mp4",  # 取消

    # "Video_20260301_153037623.avi": "interaction_20260301_073041.mp4",  # 菜单     # Gu
    # "Video_20260301_144803454.avi": "interaction_20260301_064753.mp4",  # 选择
    # "Video_20260306_152721366.avi": "interaction_20260306_072721.mp4",  # 放大
    # "Video_20260301_151942635.avi": "interaction_20260301_071948.mp4",  # 缩小
    # "Video_20260131_201556629.avi": "interaction_20260131_121548.mp4",  # 缩小
    # "Video_20260301_153434856.avi": "interaction_20260301_073435.mp4",  # 画笔
    # "Video_20260301_152459131.avi": "interaction_20260301_072503.mp4",  # 取消

    # ======================== museum ==================================
    # Luo - 2026-01-31
    "Video_20260131_151559270.avi": "interaction_20260131_071552.mp4",  # 菜单
    "Video_20260131_152410916.avi": "interaction_20260131_072412.mp4",  # 选择
    "Video_20260131_164304016.avi": "interaction_20260131_084300.mp4",  # 选择
    "Video_20260131_164745532.avi": "interaction_20260131_084732.mp4",  # 取消
    "Video_20260131_165208524.avi": "interaction_20260131_085207.mp4",  # 画笔
    "Video_20260131_165614756.avi": "interaction_20260131_085611.mp4",  # 放大
    "Video_20260131_170142792.avi": "interaction_20260131_090139.mp4",  # 缩小

    # # Gu
    # "Video_20260131_145524524.avi": "interaction_20260131_065459.mp4",  # 放大
    # "Video_20260131_150734369.avi": "interaction_20260131_070722.mp4",  # 缩小
    # "Video_20260131_170539636.avi": "interaction_20260131_090541.mp4",  # 选择
    # "Video_20260131_170919896.avi": "interaction_20260131_090917.mp4",  # 菜单
    # "Video_20260131_171253889.avi": "interaction_20260131_091249.mp4",  # 取消
    # "Video_20260131_171648040.avi": "interaction_20260131_091657.mp4",  # 画笔
    
    # Bian - 2026-03-06
    # "Video_20260306_162401599.avi": "interaction_20260306_082346.mp4",  # 放大
    # "Video_20260306_163105571.avi": "interaction_20260306_083107.mp4",  # 缩小
    # "Video_20260306_163434878.avi": "interaction_20260306_083434.mp4",  # 选择
    # "Video_20260306_164407883.avi": "interaction_20260306_084406.mp4",  # 菜单
    # "Video_20260306_164902044.avi": "interaction_20260306_084853.mp4",  # 取消
    # "Video_20260306_165839689.avi": "interaction_20260306_085830.mp4",  # 画笔
    # "Video_20260306_170449073.avi": "interaction_20260306_090441.mp4",  # 选择
}

# ==================== MediaPipe & CLIP 加载 ====================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.3)

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
clip_processor = CLIPImageProcessor.from_pretrained(CLIP_MODEL_PATH)
clip_vision = CLIPVisionModel.from_pretrained(CLIP_MODEL_PATH).to(device).eval()

def crop_hand(img_pil):
    img_np = np.array(img_pil)
    h, w = img_np.shape[:2]
    results = hands.process(img_np[..., ::-1])
    if results.multi_hand_landmarks:
        xs = [int(lm.x * w) for hand in results.multi_hand_landmarks for lm in hand.landmark]
        ys = [int(lm.y * h) for hand in results.multi_hand_landmarks for lm in hand.landmark]
        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
        pad = 0.4
        cw, ch = x2 - x1, y2 - y1
        x1, y1 = max(0, int(x1 - cw * pad)), max(0, int(y1 - ch * pad))
        x2, y2 = min(w, int(x2 + cw * pad)), min(h, int(y2 + ch * pad))
        return img_pil.crop((x1, y1, x2, y2)).resize((224, 224), Image.LANCZOS)
    return img_pil.rotate(0).resize((224, 224), Image.LANCZOS) # 回退逻辑

@torch.no_grad()
def extract_clip_feature(video_path, timestamp_ms):
    cap = cv2.VideoCapture(video_path)
    # 检查视频总时长，防止越界
    total_ms = cap.get(cv2.CAP_PROP_FRAME_COUNT) * 1000 / cap.get(cv2.CAP_PROP_FPS)
    if timestamp_ms > total_ms:
        cap.release()
        return None

    cap.set(cv2.CAP_PROP_POS_MSEC, timestamp_ms)
    ok, frame_bgr = cap.read()
    cap.release()
    if not ok: return None

    img_pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    hand_cropped = crop_hand(img_pil)
    inputs = clip_processor(images=hand_cropped, return_tensors="pt").to(device)
    outputs = clip_vision(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze(0).cpu().numpy()

# ==================== 时间计算 ====================
def get_avi_timestamp_ms(avi_path, target_utc_time):
    """
    全兼容时间对齐函数：
    无论AVI还是MP4谁先开始录制，只要动作点在AVI的时长范围内，就返回正确的毫秒偏移。
    """
    avi_filename = os.path.basename(avi_path)
    
    # 1. 提取AVI录制开始的本地时间 (精确到毫秒)
    # 示例: Video_20260301_153037623 -> 15:30:37.623
    parts = avi_filename.split('_')
    time_str = f"{parts[1]}_{parts[2].split('.')[0]}"
    avi_local_base = datetime.strptime(time_str, '%Y%m%d_%H%M%S%f')
    
    # 2. 统一转为 UTC 时间 (本地时间 - 8小时)
    avi_utc_base = avi_local_base - timedelta(hours=8)
    
    # 3. 计算【动作发生时刻】减去【视频开始时刻】的差值
    # 这个差值就是动作在视频中的相对位置
    diff_ms = (target_utc_time - avi_utc_base).total_seconds() * 1000
    
    # 4. 获取视频物理属性
    cap = cv2.VideoCapture(avi_path)
    if not cap.isOpened():
        print(f"❌ 无法打开视频文件: {avi_filename}")
        return None
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration_ms = (frame_count / fps) * 1000
    cap.release()

    # 5. 全包容判定
    # 只要 diff_ms >= 0 且不超过视频总长度，就是有效片段
    if 0 <= diff_ms <= duration_ms:
        return diff_ms
    else:
        # 只有真正超出物理范围的情况才跳过，并打印原因方便你排查
        if diff_ms < 0:
            print(f"  [跳过] 动作发生时鱼眼相机尚未开启 (提前了 {abs(diff_ms):.2f}ms)")
        else:
            print(f"  [跳过] 动作发生时鱼眼相机已经关闭 (落后了 {diff_ms - duration_ms:.2f}ms)")
        return None

# ==================== 主流程 ====================
if __name__ == "__main__":
    for avi_name, mp4_name in AVI_TO_MP4_MAP.items():
        mp4_base = os.path.splitext(mp4_name)[0]
        input_npy = os.path.join(INPUT_DATA_DIR, f"features_timestamp_{mp4_base}.npy")
        
        # 确保路径拼接正确
        video_path = os.path.join(BASE_DIR, avi_name)

        # 增加视频文件物理存在检查
        if not os.path.exists(video_path):
            print(f"❌ 跳过：视频文件不存在 -> {video_path}")
            continue

        if not os.path.exists(input_npy):
            print(f"❌ 跳过：找不到输入特征文件 {input_npy}")
            continue

        data = np.load(input_npy, allow_pickle=True).item()
        ts_list = data["approx_timestamps"]
        labels = data["labels"]
        
        valid_features, valid_labels, valid_ts = [], [], []
        debug_info = {}

        print(f"\n>>> 处理视频: {avi_name} | 原始片段数: {len(ts_list)}")

        for i, ts_str in enumerate(ts_list):
            try:
                utc_dt = datetime.fromisoformat(ts_str.replace('Z', '+00:00')).replace(tzinfo=None)
                
                # 传入完整路径 video_path
                msec_offset = get_avi_timestamp_ms(video_path, utc_dt)

                # 修复：先检查是否为 None，再进行比较
                if msec_offset is None:
                    # 具体的跳过原因已经在 get_avi_timestamp_ms 函数内部打印了
                    continue

                # 提取特征
                feat = extract_clip_feature(video_path, msec_offset)
                if feat is not None:
                    valid_features.append(feat)
                    valid_labels.append(labels[i])
                    valid_ts.append(ts_str)
                    
                    debug_info[str(len(valid_features)-1)] = {
                        "original_segment_idx": i,
                        "utc_timestamp": ts_str,
                        "avi_msec_offset": round(msec_offset, 3),
                        "label": int(labels[i])
                    }
                else:
                    print(f"  [Skip] 片段 {i} 帧读取失败")
            
            except Exception as e:
                print(f"  ❌ 处理片段 {i} 时出错: {e}")

        # --- 结果保存 ---
        if valid_features:
            # 1. 完整特征文件
            full_data = {
                "features": np.array(valid_features),
                "labels": np.array(valid_labels),
                "video_names": np.array([mp4_name] * len(valid_labels)),
                "approx_timestamps": valid_ts
            }
            np.save(os.path.join(OUTPUT_DIR, f"strong_gesture_features_{mp4_base}.npy"), full_data)

            # 2. 剔除特征的轻量文件
            meta_data = {k: v for k, v in full_data.items() if k != "features"}
            np.save(os.path.join(OUTPUT_DIR, f"metadata_strong_gesture_{mp4_base}.npy"), meta_data)

            # 3. 调试 JSON
            with open(os.path.join(OUTPUT_DIR, f"debug_strong_gesture_{mp4_base}.json"), 'w') as f:
                json.dump(debug_info, f, indent=4)
            
            print(f"  ✅ 成功保存！有效片段: {len(valid_features)}/{len(ts_list)}")
        else:
            print("  ⚠️ 该视频未提取到任何有效片段")

    print(f"\n🎉 强手势提取完成！输出目录: {OUTPUT_DIR}")
