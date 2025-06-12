import os
import zipfile
import gdown
import streamlit as st

def download_and_extract_vector_dbs(verbose: bool = False):
    """Streamlit secrets 기반으로 Google Drive에서 벡터 DB 다운로드 및 압축 해제"""

    # secrets.toml 에서 ID 가져오기
    legal_gdrive_id = st.secrets["vector_db"]["legal_gdrive_id"]
    news_gdrive_id = st.secrets["vector_db"]["news_gdrive_id"]

    files_to_download = [
        {
            "filename": "chroma_db_law_real_final.zip",
            "extract_dir": "chroma_db_law_real_final",
            "gdrive_id": legal_gdrive_id
        },
        {
            "filename": "ja_chroma_db.zip",
            "extract_dir": "ja_chroma_db",
            "gdrive_id": news_gdrive_id
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
                print(f"[INFO] Downloading '{zip_path}'...")
            gdown.download(url, zip_path, quiet=not verbose)

            if verbose:
                print(f"[INFO] Extracting '{zip_path}'...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_path)

            os.remove(zip_path)
            if verbose:
                print(f"[INFO] Done: {extract_path}")

        except Exception as e:
            print(f"[ERROR] Failed to process '{zip_path}': {e}")
