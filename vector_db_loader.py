# utils/vector_db_loader.py

import gdown
import zipfile
import os

def download_and_extract_vector_dbs(verbose: bool = False):
    """Google Drive에서 여러 개의 벡터 DB zip 파일을 다운로드하고 압축 해제"""
    
    files_to_download = [
        {
            "filename": "chroma_db_law_real_final.zip",
            "extract_dir": "chroma_db_law_real_final",
            "gdrive_id": "1gp5h0QScWB3wcsbs4i12ny1wEMY_HAqX"
        },
        {
            "filename": "ja_chroma_db.zip",
            "extract_dir": "ja_chroma_db",
            "gdrive_id": "1dU9TLAPMg-Q8DLQjZM38CC-TsK477dSO"
        }
    ]

    for file_info in files_to_download:
        zip_path = file_info["filename"]
        extract_path = file_info["extract_dir"]
        gdrive_id = file_info["gdrive_id"]

        if os.path.exists(extract_path):
            if verbose:
                print(f"[INFO] '{extract_path}' already exists. Skipping download.")
            continue

        try:
            url = f"https://drive.google.com/uc?id={gdrive_id}"
            if verbose:
                print(f"[INFO] Downloading '{zip_path}' from Google Drive...")
            gdown.download(url, zip_path, quiet=not verbose)

            if verbose:
                print(f"[INFO] Extracting '{zip_path}'...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_path)

            os.remove(zip_path)
            if verbose:
                print(f"[INFO] Extracted and cleaned up '{zip_path}'")

        except Exception as e:
            print(f"[ERROR] Failed to download or extract '{zip_path}': {str(e)}")
