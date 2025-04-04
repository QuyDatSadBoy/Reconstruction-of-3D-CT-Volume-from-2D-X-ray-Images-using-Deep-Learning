import os
import numpy as np
import pydicom
import pandas as pd
from tqdm import tqdm
import logging

# Thiết lập logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LIDCFilter:
    def __init__(self, input_folder, output_folder):
        """
        Khởi tạo bộ lọc LIDC
        Args:
            input_folder: Thư mục chứa dữ liệu LIDC gốc
            output_folder: Thư mục để lưu dữ liệu đã lọc
        """
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.suitable_patients = []
        self.patient_info = []
        
        # Tạo thư mục output nếu chưa tồn tại
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            
    def check_slice_thickness(self, slices):
        """Kiểm tra độ dày của lát cắt"""
        try:
            thicknesses = [float(s.SliceThickness) for s in slices]
            return np.std(thicknesses) < 0.1  # Độ dày phải đồng nhất
        except:
            return False
            
    def check_image_quality(self, slices):
        """Kiểm tra chất lượng hình ảnh"""
        try:
            # Kiểm tra kích thước pixel
            pixel_spacing = slices[0].PixelSpacing
            if not (0.5 <= float(pixel_spacing[0]) <= 2.0 and 0.5 <= float(pixel_spacing[1]) <= 2.0):
                return False
                
            # Kiểm tra số lượng lát cắt
            if not (300 <= len(slices) <= 550):
                return False
                
            return True
        except:
            return False
            
    def check_contrast(self, slices):
        """Kiểm tra độ tương phản của hình ảnh"""
        try:
            # Chuyển đổi sang HU units
            images = []
            for s in slices:
                image = s.pixel_array * s.RescaleSlope + s.RescaleIntercept
                images.append(image)
            
            images = np.array(images)
            
            # Tính toán độ tương phản
            contrast = np.std(images)
            return 100 <= contrast <= 2000  # Độ tương phản phải trong khoảng hợp lý
        except:
            return False
            
    def find_dicom_directories(self, patient_path):
        """Tìm tất cả các thư mục chứa file DICOM"""
        dicom_dirs = []
        for root, dirs, files in os.walk(patient_path):
            if any(f.endswith('.dcm') for f in files):
                dicom_dirs.append(root)
        return dicom_dirs
            
    def get_dicom_files_count(self, directory):
        """Đếm số lượng file DICOM trong một thư mục"""
        return len([f for f in os.listdir(directory) if f.endswith('.dcm')])
            
    def process_patient(self, patient_id):
        """Xử lý dữ liệu của một bệnh nhân"""
        try:
            patient_path = os.path.join(self.input_folder, patient_id)
            if not os.path.isdir(patient_path):
                return False
                
            # Tìm tất cả các thư mục chứa file DICOM
            dicom_dirs = self.find_dicom_directories(patient_path)
            if not dicom_dirs:
                return False
                
            # Chọn thư mục có nhiều file DICOM nhất
            best_dir = max(dicom_dirs, key=self.get_dicom_files_count)
            logger.info(f"Bệnh nhân {patient_id}: Chọn thư mục {best_dir} với {self.get_dicom_files_count(best_dir)} file DICOM")
                
            # Đọc các file DICOM từ thư mục được chọn
            slices = []
            for file in os.listdir(best_dir):
                if file.endswith('.dcm'):
                    try:
                        file_path = os.path.join(best_dir, file)
                        dicom = pydicom.read_file(file_path)
                        slices.append(dicom)
                    except:
                        continue
                    
            if not slices:
                return False
                
            # Sắp xếp các lát cắt theo vị trí
            slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
            
            # Kiểm tra các tiêu chí
            if not (self.check_slice_thickness(slices) and 
                   self.check_image_quality(slices) and 
                   self.check_contrast(slices)):
                return False
                
            # Lưu thông tin bệnh nhân
            patient_info = {
                'patient_id': patient_id,
                'num_slices': len(slices),
                'slice_thickness': float(slices[0].SliceThickness),
                'pixel_spacing': slices[0].PixelSpacing,
                'image_size': slices[0].pixel_array.shape,
                'selected_directory': best_dir
            }
            self.patient_info.append(patient_info)
            
            return True
            
        except Exception as e:
            logger.error(f"Lỗi khi xử lý bệnh nhân {patient_id}: {str(e)}")
            return False
            
    def filter_dataset(self):
        """Lọc toàn bộ tập dữ liệu"""
        logger.info("Bắt đầu quá trình lọc dữ liệu...")
        
        # Lấy danh sách tất cả các bệnh nhân
        patients = [d for d in os.listdir(self.input_folder) if d.startswith('LIDC-IDRI-')]
        
        # Xử lý từng bệnh nhân
        for patient_id in tqdm(patients, desc="Đang xử lý bệnh nhân"):
            if self.process_patient(patient_id):
                self.suitable_patients.append(patient_id)
                
        # Lưu kết quả
        self.save_results()
        
        logger.info(f"Hoàn thành! Đã tìm thấy {len(self.suitable_patients)} bệnh nhân phù hợp.")
        
    def save_results(self):
        """Lưu kết quả lọc"""
        # Lưu danh sách bệnh nhân phù hợp
        with open(os.path.join(self.output_folder, 'suitable_patients.txt'), 'w') as f:
            for patient_id in self.suitable_patients:
                f.write(f"{patient_id}\n")
                
        # Lưu thông tin chi tiết
        df = pd.DataFrame(self.patient_info)
        df.to_csv(os.path.join(self.output_folder, 'patient_info.csv'), index=False)
        
        # Lưu thống kê
        stats = {
            'total_patients': len(self.suitable_patients),
            'avg_slices': np.mean([p['num_slices'] for p in self.patient_info]),
            'std_slices': np.std([p['num_slices'] for p in self.patient_info]),
            'avg_thickness': np.mean([p['slice_thickness'] for p in self.patient_info]),
            'std_thickness': np.std([p['slice_thickness'] for p in self.patient_info])
        }
        
        with open(os.path.join(self.output_folder, 'filtering_stats.txt'), 'w') as f:
            for key, value in stats.items():
                f.write(f"{key}: {value}\n")

def main():
    # Đường dẫn đến dữ liệu
    input_folder = "./aritra_project/Data_LIDC/LIDC_IDRI"  # Thư mục chứa dữ liệu LIDC
    output_folder = "./aritra_project/filtered_data"  # Thư mục để lưu kết quả
    
    # Khởi tạo và chạy bộ lọc
    filter = LIDCFilter(input_folder, output_folder)
    filter.filter_dataset()

if __name__ == "__main__":
    main() 