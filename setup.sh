set -e

PYTHON_ENV_NAME="venv"
DATASET_GDRIVE_ID="17JjRsMOMn4rmKJ6DD2057TBaMSuQbw1y"
DATASET_ZIP_NAME="foggy_anything.zip"
DATASET_DIR_NAME="foggy_anything" # Tên thư mục sau khi giải nén

echo "apt-get update..."
sudo apt-get update
echo "Cài đặt python3-venv và unzip..."
sudo apt-get install -y python3-venv unzip

if [ ! -d "$PYTHON_ENV_NAME" ]; then
    echo "🐍 Tạo môi trường ảo Python tên là '$PYTHON_ENV_NAME'..."
    python3 -m venv $PYTHON_ENV_NAME
else
    echo "🐍 Môi trường ảo '$PYTHON_ENV_NAME' đã tồn tại."
fi

echo "Kích hoạt môi trường ảo..."
source $PYTHON_ENV_NAME/bin/activate

# --- Bước 3: Cài đặt các thư viện Python ---
echo "📦 Cài đặt các thư viện từ requirements.txt..."
pip install -r requirements.txt

# --- Bước 4: Tải và giải nén Dataset ---
echo "💾 Kiểm tra và tải dataset..."

if [ -d "$DATASET_DIR_NAME" ]; then
    echo "Thư mục dataset '$DATASET_DIR_NAME' đã tồn tại, bỏ qua bước tải và giải nén."
else
    if [ -f "$DATASET_ZIP_NAME" ]; then
        echo "File '$DATASET_ZIP_NAME' đã tồn tại, sẽ tiến hành giải nén."
    else
        echo "Tải dataset từ Google Drive (ID: $DATASET_GDRIVE_ID)..."
        gdown -O $DATASET_ZIP_NAME $DATASET_GDRIVE_ID
    fi
    
    echo "Giải nén dataset..."
    unzip -q $DATASET_ZIP_NAME
    echo "Đã giải nén xong vào thư mục '$DATASET_DIR_NAME'."

    echo "Xóa file ZIP '$DATASET_ZIP_NAME' để tiết kiệm dung lượng..."
    rm $DATASET_ZIP_NAME
fi

# --- Bước 5: Chạy file huấn luyện chính ---
echo "🔥 Bắt đầu chạy script huấn luyện chính (main.py)..."
python main.py