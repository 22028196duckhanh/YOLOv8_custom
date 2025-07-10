import os
import yaml
from ultralytics import YOLO
from datetime import datetime
import shutil

import settings

def create_dataset_yaml(output_dir):
    yaml_path = os.path.join(output_dir, settings.DATASET_YAML_NAME)
    
    dataset_config = {
        'path': os.path.abspath(settings.DATASET_PATH),
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'nc': settings.NC,
        'names': settings.CLASSES
    }
    
    with open(yaml_path, 'w') as f:
        yaml.dump(dataset_config, f, default_flow_style=False, sort_keys=False)
        
    print(f"✅ Đã tạo file cấu hình dataset tại: {yaml_path}")
    return yaml_path

def evaluate_model(model_path, data_yaml_path, eval_name):
    """
    Đánh giá một model trên tập test và lưu kết quả.
    
    Args:
        model_path (str): Đường dẫn đến file trọng số model (.pt).
        data_yaml_path (str): Đường dẫn đến file .yaml của dataset.
        eval_name (str): Tên thư mục để lưu kết quả đánh giá (ví dụ: 'baseline_eval').
    """
    print(f"\n🚀 Bắt đầu đánh giá model: {model_path}")
    model = YOLO(model_path)
    
    metrics = model.val(
        data=data_yaml_path,
        split='test',  # Chỉ định rõ ràng đánh giá trên tập test
        name=eval_name,
        project=settings.RUNS_DIR,
        save_json=True, # Lưu kết quả ra file a.json, rất quan trọng để xử lý sau này
        save_conf=True, # Lưu confusion matrix
    )
    
    print(f"----- Kết quả đánh giá cho '{eval_name}' -----")
    print(f"   mAP50-95: {metrics.box.map:.4f}")
    print(f"   mAP50:    {metrics.box.map50:.4f}")
    print(f"   mAP75:    {metrics.box.map75:.4f}")
    print(f"   Precision: {metrics.box.p[0]:.4f} (cho class '{settings.CLASSES[0]}')")
    print(f"   Recall:    {metrics.box.r[0]:.4f} (cho class '{settings.CLASSES[0]}')")
    
    # Đường dẫn thư mục kết quả mà ultralytics vừa tạo
    results_path = os.path.join(settings.RUNS_DIR, eval_name)
    print(f"✅ Kết quả đánh giá (confusion matrix, plots, json) đã được lưu tại: {results_path}")
    
    return metrics

def train_model(data_yaml_path, run_name):
    """
    Huấn luyện model YOLOv8 và lưu tất cả kết quả.
    
    Args:
        data_yaml_path (str): Đường dẫn đến file .yaml của dataset.
        run_name (str): Tên của lần chạy này, để tạo thư mục lưu kết quả.
    
    Returns:
        str: Đường dẫn đến trọng số tốt nhất (best.pt).
    """
    print("\n🔥 Bắt đầu quá trình huấn luyện model...")
    model = YOLO(settings.BASE_MODEL)
    
    results = model.train(
        data=data_yaml_path,
        epochs=settings.EPOCHS,
        imgsz=settings.IMG_SIZE,
        batch=settings.BATCH_SIZE,
        patience=settings.PATIENCE,
        device=settings.DEVICE,
        project=settings.RUNS_DIR,
        name=run_name
    )
    
    print("\n✅ Quá trình huấn luyện đã hoàn tất!")
    
    
    training_results_path = results.save_dir
    print(f"📂 Tất cả kết quả huấn luyện được lưu tại: {training_results_path}")
    
    best_model_path = os.path.join(training_results_path, 'weights/best.pt')
    
    if os.path.exists(best_model_path):
        print(f"🏆 Trọng số tốt nhất được lưu tại: {best_model_path}")
        return best_model_path
    else:
        print("⚠️ Không tìm thấy trọng số 'best.pt'.")
        return None

def main():
    """
    Hàm chính điều phối toàn bộ quy trình.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{settings.PROJECT_NAME}_{timestamp}"
    
    os.makedirs(settings.RUNS_DIR, exist_ok=True)
    
    # --- Bước 1: Chuẩn bị file YAML ---
    data_yaml_path = create_dataset_yaml(settings.RUNS_DIR)

    # --- Bước 2: Đánh giá Baseline (Model gốc trên dữ liệu của bạn) ---
    # evaluate_model(
    #     model_path=settings.BASE_MODEL, 
    #     data_yaml_path=data_yaml_path, 
    #     eval_name=f"baseline_eval_{timestamp}"
    # )

    # --- Bước 3: Huấn luyện model ---
    best_model_path = train_model(data_yaml_path, run_name)

    # --- Bước 4: Đánh giá lại model đã huấn luyện trên tập Test ---
    # if best_model_path:
    #     evaluate_model(
    #         model_path=best_model_path, 
    #         data_yaml_path=data_yaml_path, 
    #         eval_name=f"final_eval_{timestamp}"
    #     )
    # else:
    #     print("Bỏ qua bước đánh giá cuối cùng do không tìm thấy model đã huấn luyện.")

    print("\n🎉🎉🎉 Toàn bộ quy trình đã hoàn tất! 🎉🎉🎉")

if __name__ == '__main__':
    main()