import os
import torch
import numpy as np
import pandas as pd
import itertools
from glob import glob
from tqdm import tqdm
from pathlib import Path
from ultralytics import YOLO
from ultralytics.data.utils import check_det_dataset, img2label_paths
from ultralytics.utils.metrics import ConfusionMatrix, DetMetrics
from ensemble_boxes import weighted_boxes_fusion as wbf

# ==============================================================================
# BƯỚC 1: CẤU HÌNH PIPELINE
# ==============================================================================

# Danh sách các đường dẫn đến file model (.pt) của bạn
MODEL_PATHS = [
    'models/clear_model.pt',
    'models/light_model.pt',
    'models/medium_model.pt',
    'models/heavy_model.pt'
]

# Danh sách các đường dẫn đến file cấu hình dataset (.yaml) của bạn
DATASET_CONFIGS = [
    'data/test_clear.yaml',
    'data/test_foggy.yaml',
    'data/test_mix.yaml'
]

# --- Cấu hình cho Weighted Boxes Fusion (WBF) ---

# Các cặp trọng số bạn muốn thử nghiệm.
# Ví dụ: (0.6, 0.4) nghĩa là model 1 có trọng số 0.6, model 2 có trọng số 0.4
WEIGHT_PAIRS = [
    (0.5, 0.5),
    (0.6, 0.4),
    (0.4, 0.6),
    (0.7, 0.3),
    (0.3, 0.7),
    (0.8, 0.2),
    (0.2, 0.8)
]

# Các tham số khác của WBF
WBF_IOU_THR = 0.6      # Ngưỡng IoU để gộp các box
SKIP_BOX_THR = 0.001   # Bỏ qua các box có score thấp hơn ngưỡng này

# --- Cấu hình cho việc đánh giá ---
EVAL_CONFIG = {
    'imgsz': 640,
    'device': 'cuda:0',        # ID của GPU, hoặc 'cpu'
    'split': 'test'       # Tập dữ liệu để đánh giá ('val' hoặc 'test')
}

# Thư mục để lưu kết quả
RESULTS_DIR = 'wbf_evaluation_results'

# ==============================================================================
# BƯỚC 2: HÀM TIỆN ÍCH (Không cần chỉnh sửa)
# ==============================================================================

def load_ground_truth(label_path, img_height, img_width):
    """Tải và chuyển đổi nhãn ground truth từ file .txt"""
    if not os.path.exists(label_path):
        return np.empty((0, 5))
    
    gt = np.loadtxt(label_path, ndmin=2)
    # Chuyển đổi từ [class, x_center, y_center, width, height] -> [x1, y1, x2, y2, class]
    gt_xyxy = np.zeros_like(gt)
    gt_xyxy[:, 0] = (gt[:, 1] - gt[:, 3] / 2) * img_width
    gt_xyxy[:, 1] = (gt[:, 2] - gt[:, 4] / 2) * img_height
    gt_xyxy[:, 2] = (gt[:, 1] + gt[:, 3] / 2) * img_width
    gt_xyxy[:, 3] = (gt[:, 2] + gt[:, 4] / 2) * img_height
    gt_xyxy[:, 4] = gt[:, 0]
    return gt_xyxy

# ==============================================================================
# BƯỚC 3: CHẠY PIPELINE ĐÁNH GIÁ WBF
# ==============================================================================

def box_iou(box1, box2, eps=1e-7):
    """
    Calculate intersection over union (IoU) of box1(1, 4) to box2(n, 4).
    Box (x1, y1, x2, y2).
    """
    b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
    inter_x1 = torch.max(b1_x1, b2_x1)
    inter_y1 = torch.max(b1_y1, b2_y1)
    inter_x2 = torch.min(b1_x2, b2_x2)
    inter_y2 = torch.min(b1_y2, b2_y2)

    inter_area = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)

    box1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    box2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    union_area = box1_area + box2_area - inter_area + eps

    return inter_area / union_area

def run_wbf_pipeline():
    """
    Hàm chính để chạy toàn bộ quá trình đánh giá WBF.
    Phiên bản này tính toán mAP50-95 một cách chính xác.
    """
    print(" BẮT ĐẦU PIPELINE ĐÁNH GIÁ WBF ".center(80, '='))
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    all_results = []

    model_combinations = list(itertools.combinations(MODEL_PATHS, 2))

    print("--- Đang tải các model... ---")
    models = {path: YOLO(path) for path in MODEL_PATHS}
    print("--- Tải model hoàn tất! ---")

    for dataset_yaml in DATASET_CONFIGS:
        # ... (Phần load dataset không thay đổi) ...
        dataset_name = Path(dataset_yaml).stem
        print(f"\n{'='*20} ĐANG XỬ LÝ DATASET: {dataset_name.upper()} {'='*20}")
        try:
            data_info = check_det_dataset(dataset_yaml)
            split_key = EVAL_CONFIG['split']
            if not data_info.get(split_key):
                 print(f"Cảnh báo: Không tìm thấy split '{split_key}' trong file '{dataset_yaml}'. Bỏ qua...")
                 continue
            base_path = Path(data_info.get('path', '.'))
            image_dir_path = base_path / data_info[split_key]
            image_subdir_path = os.path.join(image_dir_path, 'images')
            if os.path.isdir(image_subdir_path): image_paths_glob = os.path.join(image_subdir_path, '*.*')
            else: image_paths_glob = os.path.join(image_dir_path, '*.*')
            image_paths = sorted(glob(image_paths_glob))
            if not image_paths:
                print(f"Cảnh báo: Không tìm thấy ảnh nào trong '{image_paths_glob}'. Bỏ qua dataset này.")
                continue
            label_paths = img2label_paths(image_paths)
            num_classes = int(data_info['nc'])
            original_names_dict = data_info['names']
            class_names = {i: name for i, name in original_names_dict.items()}
        except Exception as e:
            print(f"Lỗi khi đọc file dataset '{dataset_yaml}': {e}. Bỏ qua...")
            continue
        
        for model_path1, model_path2 in model_combinations:
            # ... (Phần model combination không thay đổi) ...
            model1_name = Path(model_path1).stem
            model2_name = Path(model_path2).stem
            model_combo_name = f"{model1_name}_&_{model2_name}"
            model1 = models[model_path1]
            model2 = models[model_path2]

            for weights in WEIGHT_PAIRS:
                print(f"\n--- Đang đánh giá: {model_combo_name} | Weights: {weights} | Dataset: {dataset_name} ---")

                stats_tp = []
                stats_conf = []
                stats_pred_cls = []
                stats_target_cls = []
                
                # =================================================================
                # <<< THAY ĐỔI 1: ĐỊNH NGHĨA 10 NGƯỠNG IoU CHO mAP50-95 >>>
                # =================================================================
                iou_thresholds = torch.linspace(0.5, 0.95, 10, device=EVAL_CONFIG['device'])

                for i, img_path in enumerate(tqdm(image_paths, desc=f"W: {weights}")):
                    
                    # ... (Phần predict và WBF không thay đổi) ...
                    preds1 = model1.predict(img_path, imgsz=EVAL_CONFIG['imgsz'], device=EVAL_CONFIG['device'], verbose=False)
                    preds2 = model2.predict(img_path, imgsz=EVAL_CONFIG['imgsz'], device=EVAL_CONFIG['device'], verbose=False)
                    boxes_list, scores_list, labels_list = [], [], []
                    img_height, img_width = preds1[0].orig_shape
                    p1_boxes = preds1[0].boxes
                    if len(p1_boxes) > 0: boxes_list.append(p1_boxes.xyxyn.cpu().numpy()); scores_list.append(p1_boxes.conf.cpu().numpy()); labels_list.append(p1_boxes.cls.cpu().numpy())
                    else: boxes_list.append(np.empty((0, 4))); scores_list.append(np.empty(0)); labels_list.append(np.empty(0))
                    p2_boxes = preds2[0].boxes
                    if len(p2_boxes) > 0: boxes_list.append(p2_boxes.xyxyn.cpu().numpy()); scores_list.append(p2_boxes.conf.cpu().numpy()); labels_list.append(p2_boxes.cls.cpu().numpy())
                    else: boxes_list.append(np.empty((0, 4))); scores_list.append(np.empty(0)); labels_list.append(np.empty(0))
                    boxes_wbf, scores_wbf, labels_wbf = wbf(boxes_list, scores_list, labels_list, weights=list(weights), iou_thr=WBF_IOU_THR, skip_box_thr=SKIP_BOX_THR)
                    detections = torch.zeros((len(boxes_wbf), 6), device=EVAL_CONFIG['device'])
                    if len(boxes_wbf) > 0:
                        boxes_pixel = boxes_wbf.copy(); boxes_pixel[:, [0, 2]] *= img_width; boxes_pixel[:, [1, 3]] *= img_height
                        detections[:, :4] = torch.from_numpy(boxes_pixel); detections[:, 4] = torch.from_numpy(scores_wbf); detections[:, 5] = torch.from_numpy(labels_wbf)
                    gt_xyxyc = load_ground_truth(label_paths[i], img_height, img_width)
                    labels = torch.zeros((gt_xyxyc.shape[0], 5), device=EVAL_CONFIG['device'])
                    if gt_xyxyc.shape[0] > 0: labels[:, 0] = torch.from_numpy(gt_xyxyc[:, 4]); labels[:, 1:] = torch.from_numpy(gt_xyxyc[:, :4])

                    gt_classes = labels[:, 0].int()
                    gt_bboxes = labels[:, 1:]
                    
                    if detections.shape[0] == 0:
                        if gt_classes.shape[0] > 0:
                            stats_tp.extend([[0] * 10] * gt_classes.shape[0])
                            stats_conf.extend([0] * gt_classes.shape[0])
                            stats_pred_cls.extend([0] * gt_classes.shape[0])
                            stats_target_cls.extend(gt_classes.cpu().tolist())
                        continue

                    if gt_classes.shape[0] == 0:
                        for det in detections:
                            stats_tp.append([0] * 10)
                            stats_conf.append(det[4].cpu().item())
                            stats_pred_cls.append(det[5].int().cpu().item())
                            stats_target_cls.append(0) # Dummy
                        continue
                        
                    # =================================================================
                    # <<< THAY ĐỔI 2: LOGIC KHỚP MỚI ĐỂ TÍNH mAP50-95 >>>
                    # =================================================================
                    
                    # Ma trận để theo dõi GT nào đã được khớp bởi dự đoán nào
                    # Shape: (num_detections, num_gt)
                    matches = torch.zeros(detections.shape[0], gt_bboxes.shape[0], device=EVAL_CONFIG['device'])
                    
                    for det_idx, det in enumerate(detections):
                        # Tìm các GT có cùng class
                        gt_indices_for_cls = torch.where(gt_classes == det[5].int())[0]
                        if len(gt_indices_for_cls) > 0:
                            # Tính IoU giữa dự đoán này và tất cả GT cùng class
                            ious = box_iou(det[:4].unsqueeze(0), gt_bboxes[gt_indices_for_cls]).squeeze(-1)
                            # Gán IoU vào đúng vị trí trong ma trận matches
                            matches[det_idx, gt_indices_for_cls] = ious
                    
                    # Sắp xếp các cặp khớp theo IoU giảm dần, chỉ lấy những cặp có IoU > 0.5
                    matches_filtered = matches[matches > iou_thresholds[0]]
                    if matches_filtered.shape[0] > 0:
                        # Lấy index của các cặp khớp
                        i, j = (matches > iou_thresholds[0]).nonzero(as_tuple=False).T
                        # Sắp xếp các cặp theo IoU giảm dần
                        # Điều này đảm bảo GT sẽ được khớp với dự đoán có IoU tốt nhất
                        sort_idx = matches[i, j].argsort(descending=True)
                        i, j = i[sort_idx], j[sort_idx]
                        
                        # Giữ lại cặp khớp duy nhất cho mỗi GT
                        # Nếu một GT được khớp bởi nhiều dự đoán, chỉ giữ lại cặp có IoU cao nhất
                        unique_j, count_j = j.unique(return_counts=True)
                        if (count_j > 1).any():
                           # Lấy index của cặp khớp đầu tiên (IoU cao nhất) cho mỗi GT
                           first_matches = np.array([np.where(j.cpu().numpy() == uj.cpu().numpy())[0][0] for uj in unique_j])
                           i, j = i[first_matches], j[first_matches]

                        # Giờ `i` là index của detection, `j` là index của GT đã khớp
                        final_matches = torch.stack([i, j], dim=1)
                    else:
                        final_matches = torch.empty((0, 2), dtype=torch.long)

                    # Tạo một mảng TP cho tất cả các dự đoán, ban đầu tất cả là FP
                    tp_for_img = torch.zeros(detections.shape[0], 10, device=EVAL_CONFIG['device'])

                    if final_matches.shape[0] > 0:
                        matched_det_indices = final_matches[:, 0]
                        matched_gt_indices = final_matches[:, 1]
                        
                        # Lấy IoU của các cặp đã khớp
                        matched_ious = matches[matched_det_indices, matched_gt_indices]
                        
                        # So sánh IoU này với 10 ngưỡng để tạo mảng TP
                        # (N, 1) > (1, 10) -> (N, 10)
                        tp_for_img[matched_det_indices] = (matched_ious.unsqueeze(1) >= iou_thresholds).float()
                    
                    # Thêm kết quả của ảnh này vào danh sách tổng
                    stats_tp.extend(tp_for_img.cpu().tolist())
                    stats_conf.extend(detections[:, 4].cpu().tolist())
                    stats_pred_cls.extend(detections[:, 5].int().cpu().tolist())
                    # Với mỗi dự đoán, chúng ta cần một GT tương ứng
                    # Nếu là TP, dùng class của GT đã khớp
                    # Nếu là FP, dùng class của chính nó (để xác định đúng class khi tính AP)
                    target_cls_for_img = torch.zeros_like(detections[:, 5].int())
                    if final_matches.shape[0] > 0:
                        target_cls_for_img[matched_det_indices] = gt_classes[matched_gt_indices]
                    stats_target_cls.extend(target_cls_for_img.cpu().tolist())

                    # Xử lý các GT không được khớp (FN)
                    unmatched_gt_mask = torch.ones(gt_bboxes.shape[0], dtype=torch.bool)
                    if final_matches.shape[0] > 0:
                        unmatched_gt_mask[final_matches[:, 1]] = False
                    
                    unmatched_gt_indices = torch.where(unmatched_gt_mask)[0]
                    for gt_idx in unmatched_gt_indices:
                        stats_tp.append([0] * 10)
                        stats_conf.append(0.0)
                        stats_pred_cls.append(0) # Dummy
                        stats_target_cls.append(gt_classes[gt_idx].cpu().item())

                # ... (Phần cuối cùng để process metrics không thay đổi) ...
                metrics = DetMetrics(names=class_names)
                tp_array = np.array(stats_tp)
                conf_array = np.array(stats_conf)
                pred_cls_array = np.array(stats_pred_cls)
                target_cls_array = np.array(stats_target_cls)
                # Đảm bảo TP là 2D
                if tp_array.ndim == 1: tp_array = tp_array[:, np.newaxis]
                if tp_array.shape[1] == 1 and tp_array.shape[1] < len(iou_thresholds): tp_array = np.tile(tp_array, (1, len(iou_thresholds)))
                metrics.process(tp=tp_array, conf=conf_array, pred_cls=pred_cls_array, target_cls=target_cls_array)
                stats = metrics.results_dict
                
                result_entry = {
                    'Model Combination': model_combo_name, 'Weights': f"{weights[0]}:{weights[1]}", 'Dataset': dataset_name,
                    'mAP50-95': round(stats.get('metrics/mAP50-95(B)', 0), 4),
                    'mAP50': round(stats.get('metrics/mAP50(B)', 0), 4),
                    'Precision': round(stats.get('metrics/precision(B)', 0), 4),
                    'Recall': round(stats.get('metrics/recall(B)', 0), 4),
                }
                all_results.append(result_entry)

    # ... (Phần tổng kết và lưu file không thay đổi) ...
    if not all_results: print("\nKhông có kết quả nào để tổng hợp. Vui lòng kiểm tra lại cấu hình."); return
    print("\n" + " TỔNG KẾT KẾT QUẢ ĐÁNH GIÁ WBF ".center(80, '='))
    results_df = pd.DataFrame(all_results)
    results_df.sort_values(by=['Dataset', 'Model Combination', 'mAP50'], ascending=[True, True, False], inplace=True)
    print(results_df.to_string())
    summary_csv_path = os.path.join(RESULTS_DIR, 'wbf_evaluation_summary.csv')
    results_df.to_csv(summary_csv_path, index=False, encoding='utf-8-sig')
    print("\n" + "="*80)
    print(f"Bảng tổng kết đã được lưu tại: {summary_csv_path}")
    print(" KẾT THÚC PIPELINE ".center(80, '='))

if __name__ == '__main__':
    run_wbf_pipeline()