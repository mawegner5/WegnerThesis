import zipfile
import os

# Paths
zip_file_path = '/remote_home/WegnerThesis/animals_with_attributes/AwA2-data.zip'
output_dir = '/remote_home/WegnerThesis/animals_with_attributes/'

# Unzipping function
def unzip_file(zip_path, extract_to):
    if not os.path.exists(zip_path):
        print(f"[Error] Zip file does not exist at: {zip_path}")
        return
    
    if not os.path.exists(extract_to):
        os.makedirs(extract_to)
        print(f"[Info] Created output directory: {extract_to}")
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
            print(f"[Success] Files extracted to: {extract_to}")
    except zipfile.BadZipFile:
        print(f"[Error] File is not a valid zip file: {zip_path}")
    except Exception as e:
        print(f"[Error] An unexpected error occurred: {e}")

# Run the unzip function
if __name__ == "__main__":
    unzip_file(zip_file_path, output_dir)
