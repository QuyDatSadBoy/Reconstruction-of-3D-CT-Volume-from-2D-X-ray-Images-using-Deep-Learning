#!/bin/bash

echo "===== Thiết lập môi trường cho xử lý dữ liệu CT trên Kaggle ====="

# Hiển thị thông tin GPU
echo "Kiểm tra GPU..."
nvidia-smi

# Hiển thị thông tin Python
echo "Phiên bản Python:"
python --version

# Cài đặt các gói phụ thuộc
echo "Cài đặt các thư viện từ requirements.txt..."
pip install -r requirements.txt

# Kiểm tra thư viện PyTorch
echo "Kiểm tra PyTorch và CUDA:"
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'N/A'); print('Số lượng GPU:', torch.cuda.device_count() if torch.cuda.is_available() else 0)"

# Kiểm tra thư viện xử lý ảnh
echo "Kiểm tra scikit-image:"
python -c "import skimage; print('scikit-image version:', skimage.__version__)"

# Kiểm tra thư viện đọc DICOM
echo "Kiểm tra pydicom:"
python -c "import pydicom; print('pydicom version:', pydicom.__version__)"

echo "===== Thiết lập hoàn tất ====="

# Tạo thư mục đầu ra nếu chưa tồn tại
echo "Tạo các thư mục cần thiết..."
mkdir -p ./aritra_project/dataset

# Hướng dẫn sử dụng
echo "
Hướng dẫn chạy chương trình:
1. Đảm bảo bạn đã tải dữ liệu DICOM vào thư mục ./aritra_project/filtered_dataset/
2. Chạy lệnh sau để bắt đầu xử lý: python data_generation_kaggle_optimized.py
3. Kết quả sẽ được lưu trong thư mục ./aritra_project/dataset/
" 