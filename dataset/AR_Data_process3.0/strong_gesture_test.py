#!/usr/bin/env python3
import os
import numpy as np
import torch
from transformers import CLIPImageProcessor, CLIPVisionModel
import cv2
from PIL import Image
import mediapipe as mp
import json
from tqdm import tqdm
from datetime import datetime, timedelta, timezone

# ==================== 1. 路径配置 ====================
BASE_DIR = r"D:\AR_dataset\dataset"
# get_timestamp.py 生成的文件在这里
INPUT_DATA_DIR = r"E:\AR_data_process3.0\data" 
# 结果保存目录 (特征和图片都存这里)
OUTPUT_DIR = r"E:\AR_data_process3.0\data\clean_features"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CLIP_MODEL_PATH = r"E:\AR_data_process3.0\models\clip_teacher_model"

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
    # "Video_20260227_213001434.avi": "interaction_20260227_132951.mp4",  # 选择
    # "Video_20260227_213404452.avi": "interaction_20260227_133408.mp4",  # 放大
    # "Video_20260131_194205407.avi": "interaction_20260131_114156.mp4",  # 缩小
    # "Video_20260131_195202906.avi": "interaction_20260131_115150.mp4",  # 画笔
    # "Video_20260131_194854095.avi": "interaction_20260131_114852.mp4",  # 取消

    # "Video_20260301_153037623.avi": "interaction_20260301_073041.mp4",  # 菜单     # Gu
    # "Video_20260301_144803454.avi": "interaction_20260301_064753.mp4",  # 选择
    # "Video_20260306_152721366.avi": "interaction_20260306_072721.mp4",  # 放大
    # "Video_20260301_151942635.avi": "interaction_20260301_071948.mp4",  # 缩小
    # "Video_20260131_201556629.avi": "interaction_20260131_121548.mp4",  # 缩小
    # "Video_20260301_153434856.avi": "interaction_20260301_073435.mp4",  # 画笔
    # "Video_20260301_152459131.avi": "interaction_20260301_072503.mp4",  # 取消

    # ======================== museum ==================================
    # Luo - 2026-01-31
    # "Video_20260131_151559270.avi": "interaction_20260131_071552.mp4",  # 菜单
    # "Video_20260131_152410916.avi": "interaction_20260131_072412.mp4",  # 选择
    # "Video_20260131_164304016.avi": "interaction_20260131_084300.mp4",  # 选择
    # "Video_20260131_164745532.avi": "interaction_20260131_084732.mp4",  # 取消
    # "Video_20260131_165208524.avi": "interaction_20260131_085207.mp4",  # 画笔
    # "Video_20260131_165614756.avi": "interaction_20260131_085611.mp4",  # 放大
    # "Video_20260131_170142792.avi": "interaction_20260131_090139.mp4",  # 缩小

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

# 序列采样配置
SEQ_LEN = 15                 # 序列长度
HALF_WINDOW_MS = 750        # 半窗口时长（0.75秒）

# ==================== 2. MediaPipe & CLIP 加载 ====================
mp_hands = mp.solutions.hands
# 这里的 static_image_mode=True 保证对单帧的准确裁剪，后面跑序列时会重新处理
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.3)

# 显卡 4090
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(f"🚀 [设备] 正在使用: {device}")
clip_processor = CLIPImageProcessor.from_pretrained(CLIP_MODEL_PATH)
clip_vision = CLIPVisionModel.from_pretrained(CLIP_MODEL_PATH).to(device).eval()

# ==================== 3. 核心工具函数 ====================

def crop_hand(img_pil):
    """
    使用 MediaPipe 检测并裁剪手部区域，返回 PIL 图片
    """
    img_np = np.array(img_pil)
    h, w = img_np.shape[:2]
    # MediaPipe 需要 RGB 输入，这里转为 RGB
    img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    
    if results.multi_hand_landmarks:
        xs = [int(lm.x * w) for hand in results.multi_hand_landmarks for lm in hand.landmark]
        ys = [int(lm.y * h) for hand in results.multi_hand_landmarks for lm in hand.landmark]
        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
        
        # 增加 40% Padding
        pad = 0.4
        cw, ch = x2 - x1, y2 - y1
        x1, y1 = max(0, int(x1 - cw * pad)), max(0, int(y1 - ch * pad))
        x2, y2 = min(w, int(x2 + cw * pad)), min(h, int(y2 + ch * pad))
        
        return img_pil.crop((x1, y1, x2, y2)).resize((224, 224), Image.LANCZOS)
    
    # 回退：如果不检测出手，返回旋转0度并缩放的图（防止对齐出错）
    return img_pil.rotate(0).resize((224, 224), Image.LANCZOS)

@torch.no_grad()
def extract_clip_feature_sequence(video_path, center_timestamp_ms):
    """
    重写：在中心时间点前后滑动采样 15 帧，提取序列特征和中心灰度图。
    返回: (sequence_features [15, Dim], center_hand_gray_pil)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return None, None
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    total_ms = (total_frames / fps) * 1000
    cap.release()
    
    # 1. 窗口合法性校验：确保窗口不超出视频边界
    win_start_ms = center_timestamp_ms - HALF_WINDOW_MS
    win_end_ms = center_timestamp_ms + HALF_WINDOW_MS
    
    if win_start_ms < 0 or win_end_ms > total_ms:
        # print(f"  [Skip] 时间窗口越界 (Total: {total_ms:.1f}ms)")
        return None, None
    
    # 2. 生成 15 次采样的毫秒时间轴
    # 使用 np.linspace 确保在 win_start_ms 到 win_end_ms 之间均匀采样 15 点
    seq_offsets_ms = np.linspace(win_start_ms, win_end_ms, SEQ_LEN)
    
    sequence_features = []
    center_hand_gray_pil = None
    
    # 第 8 帧（索引 7）是几何中点
    center_idx = SEQ_LEN // 2
    
    # 重开 cap 跑序列提取
    cap = cv2.VideoCapture(video_path)
    
    for i, msec in enumerate(seq_offsets_ms):
        cap.set(cv2.CAP_PROP_POS_MSEC, msec)
        ok, frame_bgr = cap.read()
        if not ok: break
        
        img_pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        hand_cropped = crop_hand(img_pil) # 裁剪出的 $224 \times 224$ PIL 图片
        
        # 3. 如果是中心帧，保存其裁剪后的灰度图
        if i == center_idx:
            # 根据项目总体要求，保存为灰度图
            center_hand_gray_pil = hand_cropped.convert("L")
            
        # 4. 提取特征 (这里必须在提取前转回 RGB)
        # 裁剪出的是单通道灰度图，这里重新转为 RGB 以输入 CLIP
        # 因为 crop_hand 里已经转过 RGB 做检测，这里 hand_cropped 本质上是 RGB
        # 如果前面 convert("L")，这里要转回 RGB
        # 为了清晰，显式处理：
        img_input_clip = hand_cropped.convert("RGB")
        
        inputs = clip_processor(images=img_input_clip, return_tensors="pt").to(device)
        outputs = clip_vision(**inputs)
        # 提取 CLS Token 特征：形状为 (Dim,)
        feat = outputs.last_hidden_state[:, 0, :].squeeze(0).cpu().numpy()
        sequence_features.append(feat)
        
    cap.release()
    
    # 5. 校验：确保成功提取了 15 帧
    if len(sequence_features) != SEQ_LEN:
        # print(f"  [Skip] 时序帧读取不足 {SEQ_LEN} 帧")
        return None, None
        
    return np.array(sequence_features), center_hand_gray_pil

def get_avi_timestamp_ms(avi_path, target_utc_time):
    """
    时间对齐函数 (保持原逻辑)
    """
    avi_filename = os.path.basename(avi_path)
    parts = avi_filename.split('_')
    time_str = f"{parts[1]}_{parts[2].split('.')[0]}"
    try:
        avi_local_base = datetime.strptime(time_str, '%Y%m%d_%H%M%S%f')
        avi_utc_base = avi_local_base - timedelta(hours=8)
    except Exception as e:
        print(f"❌ 无法从文件名提取时间: {avi_filename}, 错误: {e}")
        return None
    
    diff_ms = (target_utc_time - avi_utc_base).total_seconds() * 1000
    
    cap = cv2.VideoCapture(avi_path)
    if not cap.isOpened(): return None
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration_ms = (frame_count / fps) * 1000
    cap.release()

    if 0 <= diff_ms <= duration_ms:
        return diff_ms
    else:
        if diff_ms < 0:
            print(f"  [Skip] 鱼眼尚未开启 (提前 {abs(diff_ms):.1f}ms)")
        else:
            print(f"  [Skip] 鱼眼已关闭 (落后 {diff_ms - duration_ms:.1f}ms)")
        return None

# ==================== 4. 主流程 ====================
if __name__ == "__main__":
    for avi_name, mp4_name in AVI_TO_MP4_MAP.items():
        mp4_base = os.path.splitext(mp4_name)[0]
        input_npy = os.path.join(INPUT_DATA_DIR, f"features_timestamp_{mp4_base}.npy")
        video_path = os.path.join(BASE_DIR, avi_name)

        # 视频存在检查
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

        print(f"\n>>> 处理鱼眼视频: {avi_name} | 原始动作点数: {len(ts_list)}")

        # 准备图片保存子目录：以视频名为单位
        img_sub_dir = os.path.join(OUTPUT_DIR, f"images_center_{mp4_base}")
        os.makedirs(img_sub_dir, exist_ok=True)

        for i, ts_str in tqdm(enumerate(ts_list), total=len(ts_list), desc=" 采样"):
            try:
                # 转换 UTC ISO 为 datetime
                utc_dt = datetime.fromisoformat(ts_str.replace('Z', '+00:00')).replace(tzinfo=None)
                msec_offset = get_avi_timestamp_ms(video_path, utc_dt)

                if msec_offset is None: continue

                # 调用新函数：提取 15 帧序列特征和中心灰度帧图片
                seq_feat, center_img_gray_pil = extract_clip_feature_sequence(video_path, msec_offset)
                
                if seq_feat is not None and center_img_gray_pil is not None:
                    valid_features.append(seq_feat) # 特征形状: (15, Dim)
                    valid_labels.append(labels[i])
                    valid_ts.append(ts_str)
                    
                    # === 新增：保存中心帧灰度图片 ===
                    img_name = f"{mp4_base}_seg{i:03d}_{msec_offset:.0f}ms_label{labels[i]}.jpg"
                    img_save_path = os.path.join(img_sub_dir, img_name)
                    # 将 PIL 图片保存为灰度 JPEG
                    center_img_gray_pil.save(img_save_path, "JPEG")
                    
                    debug_info[str(len(valid_features)-1)] = {
                        "original_segment_idx": i,
                        "utc_timestamp": ts_str,
                        "avi_msec_offset_center": round(msec_offset, 3),
                        "time_window_ms": [round(msec_offset - HALF_WINDOW_MS, 3), round(msec_offset + HALF_WINDOW_MS, 3)],
                        "sequence_length": SEQ_LEN,
                        "center_frame_index": SEQ_LEN // 2,
                        "label": int(labels[i]),
                        "saved_image_path": os.path.basename(img_save_path)
                    }
                else:
                    # 这通常发生在 MediaPipe 裁剪失败（出画）或读取不足 15 帧
                    pass 
            
            except Exception as e:
                print(f"  ❌ 处理动作点 {i} 时出错: {e}")

        # --- 结果保存 ---
        if valid_features:
            # 1. 完整特征文件 (核心)
            full_data = {
                "features": np.array(valid_features),                   # (N, 15, Dim)
                "labels": np.array(valid_labels),                       # (N,)
                "video_names": np.array([mp4_name] * len(valid_labels)), # (N,)
                "approx_timestamps": valid_ts                           # (N,) Center UTC
            }
            np.save(os.path.join(OUTPUT_DIR, f"strong_gesture_features_{mp4_base}.npy"), full_data)

            # 2. 剔除特征的轻量 Metadata
            meta_data = {k: v for k, v in full_data.items() if k != "features"}
            np.save(os.path.join(OUTPUT_DIR, f"metadata_strong_gesture_{mp4_base}.npy"), meta_data)

            # 3. 详细调试 JSON
            with open(os.path.join(OUTPUT_DIR, f"debug_strong_gesture_{mp4_base}.json"), 'w') as f:
                json.dump(debug_info, f, indent=4)
            
            print(f"  ✅ 保存完成！有效片段: {len(valid_features)}/{len(ts_list)}")
            print(f"  📸 中心帧图片已存入: {img_sub_dir}")
        else:
            print("  ⚠️ 该视频未提取到任何有效时序片段")

    print(f"\n🎉 15帧强手势时序特征提取与关键帧保存完成！")
    print(f"   输出目录: {OUTPUT_DIR}")
