import numpy as np
import pydicom
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from skimage.transform import resize

# Thiết lập để sử dụng GPU nếu có
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Đặt thành "" nếu muốn chỉ sử dụng CPU

def get_pixels_hu(slices):
    """Chuyển đổi giá trị pixel thành đơn vị HU (Hounsfield Units)"""
    image = np.stack([s.pixel_array for s in slices])
    image = image.astype(np.int16)
    
    # Đặt giá trị pixel bên ngoài vùng quét thành 0
    image[image == -2000] = 0
    
    # Chuyển đổi sang đơn vị HU
    for slice_number in range(len(slices)):
        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope
        
        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)
            
        image[slice_number] += np.int16(intercept)
    
    return np.array(image, dtype=np.int16)

def resample_to_target_shape(image, target_shape=(256, 256, 256)):
    """
    Lấy mẫu lại khối dữ liệu về kích thước mục tiêu sử dụng skimage.transform.resize
    Phương pháp này sẽ tiết kiệm bộ nhớ và đảm bảo kích thước chính xác
    """
    start_time = time.time()
    print(f"Resampling từ {image.shape} sang {target_shape}")
    
    # Lấy mẫu lại với skimage.transform.resize
    resampled = resize(
        image.astype(np.float32),  # Chuyển thành float32 cho xử lý nhẹ hơn
        target_shape,
        order=1,                   # Linear interpolation
        preserve_range=True,       # Giữ nguyên khoảng giá trị
        anti_aliasing=True         # Chống răng cưa
    )
    
    end_time = time.time()
    print(f"Hoàn thành resampling trong {end_time - start_time:.2f} giây")
    print(f"Kích thước sau khi resampling: {resampled.shape}")
    
    return resampled.astype(np.float32)

def process_patient(patient_id, input_folder, output_folder, target_shape=(256, 256, 256)):
    """Xử lý dữ liệu CT của một bệnh nhân và lưu kết quả"""
    try:
        print(f"\nĐang xử lý bệnh nhân {patient_id}...")
        start_time = time.time()
        
        # Đường dẫn thư mục
        patient_path = os.path.join(input_folder, patient_id)
        ct_scan_path = os.path.join(patient_path, 'CT_scan')
        
        # Kiểm tra thư mục có tồn tại
        if not os.path.exists(ct_scan_path):
            print(f"Không tìm thấy thư mục CT_scan cho bệnh nhân {patient_id}")
            return False
        
        # Đọc tất cả các file DICOM
        dcm_files = [os.path.join(ct_scan_path, f) for f in os.listdir(ct_scan_path) 
                    if f.lower().endswith('.dcm')]
        
        if not dcm_files:
            print(f"Không tìm thấy file DICOM cho bệnh nhân {patient_id}")
            return False
            
        dcm_files.sort()
        
        # Đọc các file DICOM
        print(f"Đọc {len(dcm_files)} file DICOM...")
        dcm_slices = []
        for file_path in tqdm(dcm_files, desc="Đọc DICOM"):
            try:
                dicom = pydicom.read_file(file_path, force=True)
                dcm_slices.append(dicom)
            except Exception as e:
                print(f"Lỗi khi đọc {file_path}: {str(e)}")
                continue
        
        if not dcm_slices:
            print(f"Không đọc được file DICOM nào cho bệnh nhân {patient_id}")
            return False
        
        # Sắp xếp lát cắt theo vị trí
        try:
            dcm_slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
        except:
            # Nếu không có ImagePositionPatient, thử sắp xếp theo SliceLocation
            try:
                dcm_slices.sort(key=lambda x: float(x.SliceLocation))
            except:
                # Nếu cả hai cách đều không được, giữ nguyên thứ tự
                print("Không thể sắp xếp các lát cắt theo vị trí, giữ nguyên thứ tự")
        
        # Chuyển đổi sang đơn vị HU
        print("Chuyển đổi sang đơn vị HU...")
        patient_pixels = get_pixels_hu(dcm_slices)
        print(f"Khối dữ liệu gốc: {patient_pixels.shape}, Min: {np.min(patient_pixels)}, Max: {np.max(patient_pixels)}")
        
        # Lưu thông tin spacing gốc
        try:
            original_spacing = [float(dcm_slices[0].SliceThickness)] + [float(x) for x in dcm_slices[0].PixelSpacing]
        except:
            original_spacing = [1.0, 1.0, 1.0]  # Giá trị mặc định nếu không tìm thấy
            print("Không tìm thấy thông tin spacing, sử dụng giá trị mặc định")
        
        # Resampling
        print(f"Đang lấy mẫu lại khối dữ liệu...")
        resampled_volume = resample_to_target_shape(patient_pixels, target_shape)
        print(f"Khối dữ liệu sau khi lấy mẫu lại: {resampled_volume.shape}")
        
        # Chuẩn hóa giá trị về [0, 1]
        print("Chuẩn hóa giá trị pixel...")
        min_val = np.min(resampled_volume)
        max_val = np.max(resampled_volume)
        normalized_volume = (resampled_volume - min_val) / (max_val - min_val)
        
        # Tạo thư mục đầu ra
        patient_output_dir = os.path.join(output_folder, patient_id)
        os.makedirs(patient_output_dir, exist_ok=True)
        
        # Lưu khối CT
        output_path = os.path.join(patient_output_dir, f"{patient_id}.npy")
        print(f"Đang lưu khối CT vào {output_path}")
        np.save(output_path, normalized_volume)
        
        # Lưu thông tin cấu hình và spacing
        info = {
            'original_shape': patient_pixels.shape,
            'resampled_shape': resampled_volume.shape,
            'original_spacing': original_spacing,
            'hu_min': float(min_val),
            'hu_max': float(max_val)
        }
        
        # Lưu thông tin dưới dạng text
        with open(os.path.join(patient_output_dir, f"{patient_id}_info.txt"), 'w') as f:
            for key, value in info.items():
                f.write(f"{key}: {value}\n")
        
        # Tạo một hình ảnh mẫu để kiểm tra
        mid_slice = normalized_volume.shape[1] // 2
        plt.figure(figsize=(10, 10))
        plt.imshow(normalized_volume[:, mid_slice, :], cmap='gray')
        plt.title(f"Mid-slice of {patient_id}")
        plt.savefig(os.path.join(patient_output_dir, f"{patient_id}_sample.png"))
        plt.close()
        
        end_time = time.time()
        print(f"Hoàn thành xử lý bệnh nhân {patient_id} trong {end_time - start_time:.2f} giây")
        return True
    
    except Exception as e:
        print(f"Lỗi khi xử lý bệnh nhân {patient_id}: {str(e)}")
        return False

def main():
    # Cấu hình đường dẫn
    input_folder = './aritra_project/filtered_dataset/'  # Thư mục chứa dữ liệu đã lọc
    output_folder = './aritra_project/dataset'  # Thư mục đầu ra
    
    # Tạo thư mục đầu ra nếu chưa tồn tại
    os.makedirs(output_folder, exist_ok=True)
    
    # Liệt kê danh sách bệnh nhân
    patients = [d for d in os.listdir(input_folder) 
               if os.path.isdir(os.path.join(input_folder, d)) and d != 'copy_summary.txt']
    patients.sort()
    
    # Xử lý từng bệnh nhân một
    print(f"Tìm thấy {len(patients)} bệnh nhân để xử lý")
    
    # Lấy danh sách bệnh nhân đã xử lý (nếu có)
    processed_patients = []
    processed_file = os.path.join(output_folder, 'processed_patients.txt')
    if os.path.exists(processed_file):
        with open(processed_file, 'r') as f:
            processed_patients = [line.strip() for line in f.readlines()]
    
    # Chọn các bệnh nhân chưa xử lý
    patients_to_process = [p for p in patients if p not in processed_patients]
    print(f"Còn {len(patients_to_process)} bệnh nhân cần xử lý")
    
    # Tuỳ chỉnh kích thước mục tiêu - sử dụng 256³ để tiết kiệm bộ nhớ
    target_shape = (256, 256, 256)  # Thay đổi thành (512, 512, 512) nếu muốn độ phân giải cao hơn
    
    # Xử lý từng bệnh nhân
    success_count = 0
    for patient_id in tqdm(patients_to_process, desc="Đang xử lý bệnh nhân"):
        if process_patient(patient_id, input_folder, output_folder, target_shape):
            success_count += 1
            # Ghi nhận bệnh nhân đã xử lý thành công
            with open(processed_file, 'a') as f:
                f.write(f"{patient_id}\n")
    
    print(f"Đã xử lý thành công {success_count}/{len(patients_to_process)} bệnh nhân")
    
    # Tạo báo cáo tổng hợp
    with open(os.path.join(output_folder, 'processing_summary.txt'), 'w') as f:
        f.write(f"Tổng số bệnh nhân: {len(patients)}\n")
        f.write(f"Số bệnh nhân đã xử lý: {len(processed_patients) + success_count}\n")
        f.write(f"Số bệnh nhân xử lý thành công: {success_count}\n")
        f.write(f"Tỷ lệ thành công: {success_count/max(1, len(patients_to_process))*100:.2f}%\n")
        f.write(f"Target shape: {target_shape}\n")

if __name__ == "__main__":
    main() 