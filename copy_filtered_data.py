import os
import shutil
import pandas as pd
from tqdm import tqdm
import logging

# Thiết lập logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataCopier:
    def __init__(self, source_folder, filtered_data_folder, output_folder):
        """
        Khởi tạo bộ copy dữ liệu
        Args:
            source_folder: Thư mục chứa dữ liệu LIDC gốc
            filtered_data_folder: Thư mục chứa kết quả lọc (chứa suitable_patients.txt và patient_info.csv)
            output_folder: Thư mục để lưu dữ liệu đã lọc
        """
        self.source_folder = source_folder
        self.filtered_data_folder = filtered_data_folder
        self.output_folder = output_folder
        
        # Tạo thư mục output nếu chưa tồn tại
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            
    def read_filtered_data(self):
        """Đọc dữ liệu đã lọc"""
        try:
            # Đọc danh sách bệnh nhân phù hợp
            with open(os.path.join(self.filtered_data_folder, 'suitable_patients.txt'), 'r') as f:
                self.suitable_patients = [line.strip() for line in f.readlines()]
                
            # Đọc thông tin chi tiết
            self.patient_info = pd.read_csv(os.path.join(self.filtered_data_folder, 'patient_info.csv'))
            
            logger.info(f"Đã đọc thông tin của {len(self.suitable_patients)} bệnh nhân phù hợp")
            return True
        except Exception as e:
            logger.error(f"Lỗi khi đọc dữ liệu đã lọc: {str(e)}")
            return False
            
    def copy_patient_data(self, patient_id):
        """Copy dữ liệu của một bệnh nhân"""
        try:
            # Tạo thư mục cho bệnh nhân trong output
            patient_output_dir = os.path.join(self.output_folder, patient_id)
            if not os.path.exists(patient_output_dir):
                os.makedirs(patient_output_dir)
                
            # Lấy thông tin về thư mục được chọn
            patient_data = self.patient_info[self.patient_info['patient_id'] == patient_id].iloc[0]
            selected_dir = patient_data['selected_directory']
            
            # Copy toàn bộ thư mục chứa file DICOM
            shutil.copytree(selected_dir, os.path.join(patient_output_dir, 'CT_scan'))
            
            # Copy metadata nếu có
            metadata_file = os.path.join(self.source_folder, patient_id, 'metadata.csv')
            if os.path.exists(metadata_file):
                shutil.copy2(metadata_file, patient_output_dir)
                
            logger.info(f"Đã copy dữ liệu của bệnh nhân {patient_id}")
            return True
        except Exception as e:
            logger.error(f"Lỗi khi copy dữ liệu của bệnh nhân {patient_id}: {str(e)}")
            return False
            
    def copy_filtered_dataset(self):
        """Copy toàn bộ tập dữ liệu đã lọc"""
        logger.info("Bắt đầu quá trình copy dữ liệu...")
        
        # Đọc dữ liệu đã lọc
        if not self.read_filtered_data():
            return
            
        # Copy dữ liệu của từng bệnh nhân
        success_count = 0
        for patient_id in tqdm(self.suitable_patients, desc="Đang copy dữ liệu"):
            if self.copy_patient_data(patient_id):
                success_count += 1
                
        logger.info(f"Hoàn thành! Đã copy thành công {success_count}/{len(self.suitable_patients)} bệnh nhân")
        
        # Tạo file thống kê
        self.create_summary()
        
    def create_summary(self):
        """Tạo file tổng kết"""
        summary = {
            'total_patients': len(self.suitable_patients),
            'successfully_copied': len([d for d in os.listdir(self.output_folder) if os.path.isdir(os.path.join(self.output_folder, d))]),
            'source_folder': self.source_folder,
            'output_folder': self.output_folder
        }
        
        with open(os.path.join(self.output_folder, 'copy_summary.txt'), 'w') as f:
            for key, value in summary.items():
                f.write(f"{key}: {value}\n")

def main():
    # Đường dẫn đến các thư mục
    source_folder = "./aritra_project/Data_LIDC/LIDC_IDRI"  # Thư mục chứa dữ liệu LIDC gốc
    filtered_data_folder = "./aritra_project/filtered_data"  # Thư mục chứa kết quả lọc
    output_folder = "./aritra_project/filtered_dataset"  # Thư mục để lưu dữ liệu đã lọc
    
    # Khởi tạo và chạy bộ copy
    copier = DataCopier(source_folder, filtered_data_folder, output_folder)
    copier.copy_filtered_dataset()

if __name__ == "__main__":
    main() 