# Tạo ảnh DiffDRR từ CT và Huấn luyện mô hình 3D

Dự án này giúp tạo ảnh DiffDRR (Digitally Reconstructed Radiographs) từ dữ liệu CT và huấn luyện mô hình 3D từ các ảnh này.

## Yêu cầu

```bash
# Cài đặt các thư viện cần thiết
pip install diffdrr pydicom numpy matplotlib tqdm torchio scikit-image torch torchvision
```

## Cấu trúc dữ liệu

Dữ liệu đầu vào nên có cấu trúc như sau:

```
filtered_dataset/
├── LIDC-IDRI-XXXX/
│   └── CT_scan/
│       ├── 1-XXX.dcm
│       └── ...
├── LIDC-IDRI-YYYY/
└── ...
```

## Cách sử dụng

### 1. Tạo ảnh DiffDRR từ dữ liệu CT

Script `create_diffdrr_images.py` tạo ảnh DiffDRR từ mặt trước (AP view) của dữ liệu CT và lưu thành file NPY.

```bash
python create_diffdrr_images.py --input_dir aritra_project/filtered_dataset --output_dir diffdrr_output
```

Tham số:
- `--input_dir`: Thư mục chứa dữ liệu CT (mặc định: `aritra_project/filtered_dataset`)
- `--output_dir`: Thư mục lưu kết quả (mặc định: `diffdrr_output`)
- `--num_samples`: Số lượng mẫu cần xử lý (mặc định: tất cả)

Kết quả sẽ được lưu trong thư mục sau:
```
diffdrr_output/
├── npy/                   # Ảnh DRR dạng NPY
│   ├── LIDC-IDRI-XXXX_ap.npy
│   └── ...
└── visualization/         # Ảnh để hiển thị
    ├── LIDC-IDRI-XXXX_ap.png
    └── ...
```

### 2. Chuẩn bị dữ liệu cho Kaggle

Script `prepare_kaggle_dataset.py` chuẩn bị dữ liệu để upload lên Kaggle.

```bash
python prepare_kaggle_dataset.py --input_dir diffdrr_output --output_dir diffdrr_kaggle_dataset
```

Tham số:
- `--input_dir`: Thư mục chứa kết quả từ bước 1 (mặc định: `diffdrr_output`)
- `--output_dir`: Thư mục đầu ra để lưu dataset đã tổ chức (mặc định: `diffdrr_kaggle_dataset`)

Kết quả sẽ tạo ra một dataset chuẩn cho Kaggle với cấu trúc:
```
diffdrr_kaggle_dataset/
├── npy/
├── visualization/
├── metadata.json
├── dataset_thumbnail.png
└── README.md
```

### 3. Huấn luyện mô hình trên Kaggle

Script `kaggle_train_model.py` huấn luyện mô hình 3D từ ảnh DiffDRR trên Kaggle.

1. Upload dataset lên Kaggle (từ thư mục `diffdrr_kaggle_dataset`)
2. Tạo notebook mới trên Kaggle và sử dụng dataset đã upload
3. Upload file `kaggle_train_model.py` vào notebook
4. Chạy lệnh sau trong notebook:

```python
!python kaggle_train_model.py --data_dir /kaggle/input/diffdrr-ap-views --output_dir /kaggle/working/output
```

Tham số:
- `--data_dir`: Đường dẫn tới dataset trên Kaggle
- `--output_dir`: Thư mục lưu kết quả trên Kaggle
- `--batch_size`: Kích thước batch (mặc định: 16)
- `--num_epochs`: Số epoch huấn luyện (mặc định: 100)
- `--learning_rate`: Tốc độ học ban đầu (mặc định: 0.001)

## Chi tiết kỹ thuật

### Tạo ảnh DiffDRR

- Sử dụng phương pháp Siddon để mô phỏng xạ tia X qua volume CT
- Góc chụp là mặt trước (Anterior-Posterior view)
- Kích thước ảnh mặc định: 512x512 pixels
- Khoảng cách từ nguồn đến bộ phát hiện (SDD): 1000.0mm

### Mô hình U-Net

- Mô hình U-Net 2D để học biểu diễn từ ảnh DRR
- Encoder: Conv2D + BatchNorm + ReLU + MaxPool
- Decoder: ConvTranspose2D + BatchNorm + ReLU
- Loss function: MSE (Mean Square Error)
- Optimizer: Adam

## Lưu ý

- Script sẽ tự động phát hiện và sử dụng GPU nếu có
- Quá trình tạo DRR có thể mất nhiều thời gian tùy thuộc vào số lượng bệnh nhân
- Mô hình được lưu sau mỗi epoch nếu validation loss giảm

## Trích dẫn

Nếu bạn sử dụng mã nguồn này, vui lòng trích dẫn:

```
@inproceedings{gopalakrishnan2022fast,
  title={Fast auto-differentiable digitally reconstructed radiographs for solving inverse problems in intraoperative imaging},
  author={Gopalakrishnan, Vivek and Golland, Polina},
  booktitle={Workshop on Clinical Image-Based Procedures},
  pages={1--11},
  year={2022},
  organization={Springer}
}
```

## Giấy phép

MIT







