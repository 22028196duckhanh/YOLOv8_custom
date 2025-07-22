import os
import pandas as pd
from ultralytics import YOLO
from pathlib import Path

# ==============================================================================
# BƯỚC 1: CẤU HÌNH PIPELINE
# Vui lòng cập nhật các đường dẫn này cho phù hợp với dự án của bạn
# ==============================================================================

# Danh sách các đường dẫn đến file model (.pt) của bạn
MODEL_PATHS = [
    'models/clear_model.pt',
    'models/foggy_model.pt',
    'models/mix_model.pt'
]


DATASET_CONFIGS = [
    'data/test_clear.yaml',
    'data/test_foggy.yaml',
    'data/test_mix.yaml'
]

EVAL_CONFIG = {
    'imgsz': 640,       
    'batch': 8,        
    'split': 'test',    
    'conf': 0.001,      
    'iou': 0.6,         
    'device': '0'      
}

# Thư mục để lưu tất cả kết quả từ ultralytics (confusion matrix, P-R curve, etc.)
RESULTS_DIR = 'evaluation_results'

# ==============================================================================
# BƯỚC 2: CHẠY PIPELINE ĐÁNH GIÁ (Không cần chỉnh sửa phần dưới)
# ==============================================================================

def run_evaluation_pipeline():
    """
    Hàm chính để chạy toàn bộ quá trình đánh giá các model trên các dataset.
    """
    print(" BẮT ĐẦU PIPELINE ĐÁNH GIÁ ".center(80, '='))
    
    # Tạo thư mục chứa kết quả nếu chưa tồn tại
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # List để lưu kết quả của mỗi lần chạy
    all_results = []

    # Vòng lặp qua từng model
    for model_path in MODEL_PATHS:
        if not os.path.exists(model_path):
            print(f"Cảnh báo: Không tìm thấy model tại '{model_path}'. Bỏ qua...")
            continue
            
        # Tải model
        print(f"\n--- Đang tải model: {model_path} ---")
        model = YOLO(model_path)
        model_name = Path(model_path).stem # Lấy tên file model không có extension

        # Vòng lặp qua từng dataset
        for dataset_yaml in DATASET_CONFIGS:
            if not os.path.exists(dataset_yaml):
                print(f"Cảnh báo: Không tìm thấy file config dataset '{dataset_yaml}'. Bỏ qua...")
                continue
            
            dataset_name = Path(dataset_yaml).parent.name # Lấy tên thư mục chứa dataset
            
            print(f"\n>>> Đang đánh giá model '{model_name}' trên dataset '{dataset_name}'...")
            
            # Tạo một tên duy nhất cho lần chạy này để lưu kết quả chi tiết
            run_name = f"{model_name}_on_{dataset_name}"
            
            # Chạy quá trình đánh giá (validation)
            try:
                metrics = model.val(
                    data=dataset_yaml,
                    split=EVAL_CONFIG['split'],
                    imgsz=EVAL_CONFIG['imgsz'],
                    batch=EVAL_CONFIG['batch'],
                    conf=EVAL_CONFIG['conf'],
                    iou=EVAL_CONFIG['iou'],
                    device=EVAL_CONFIG['device'],
                    project=RESULTS_DIR, # Lưu kết quả vào thư mục chính
                    name=run_name        # Tạo thư mục con cho lần chạy này
                )
                
                # Trích xuất các chỉ số quan trọng
                # metrics.box.map: mAP50-95
                # metrics.box.map50: mAP50
                # metrics.box.map75: mAP75
                # metrics.box.p: Precision (list, 1 giá trị cho mỗi class)
                # metrics.box.r: Recall (list, 1 giá trị cho mỗi class)
                
                # Vì bạn có 2 class, chúng ta sẽ lấy precision và recall cho cả 2 class
                result_entry = {
                    'Model': model_name,
                    'Dataset': dataset_name,
                    'mAP50-95': round(metrics.box.map, 4),
                    'mAP50': round(metrics.box.map50, 4),
                    'mAP75': round(metrics.box.map75, 4),
                    'Precision (Class 0)': round(metrics.box.p[0], 4),
                    'Recall (Class 0)': round(metrics.box.r[0], 4),
                    'Precision (Class 1)': round(metrics.box.p[1], 4),
                    'Recall (Class 1)': round(metrics.box.r[1], 4),
                }
                all_results.append(result_entry)
                
                print(f">>> Hoàn thành! Kết quả chi tiết được lưu tại: {RESULTS_DIR}/{run_name}")

            except Exception as e:
                print(f"!!! Lỗi khi đánh giá model '{model_name}' trên dataset '{dataset_name}': {e}")

    # ==============================================================================
    # BƯỚC 3: TỔNG HỢP VÀ HIỂN THỊ KẾT QUẢ CUỐI
    # ==============================================================================

    if not all_results:
        print("\nKhông có kết quả nào để tổng hợp. Vui lòng kiểm tra lại cấu hình.")
        return

    print("\n" + " TỔNG KẾT KẾT QUẢ ĐÁNH GIÁ ".center(80, '='))
    
    # Tạo DataFrame từ list kết quả
    results_df = pd.DataFrame(all_results)
    
    # Hiển thị bảng kết quả trên terminal
    print(results_df.to_string())
    
    # Lưu bảng kết quả vào file CSV để dễ dàng phân tích sau này
    summary_csv_path = os.path.join(RESULTS_DIR, 'evaluation_summary.csv')
    results_df.to_csv(summary_csv_path, index=False, encoding='utf-8-sig')
    
    print("\n" + "="*80)
    print(f"Bảng tổng kết đã được lưu tại: {summary_csv_path}")
    print(" KẾT THÚC PIPELINE ".center(80, '='))

if __name__ == '__main__':
    run_evaluation_pipeline()