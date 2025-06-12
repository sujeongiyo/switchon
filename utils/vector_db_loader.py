# utils/vector_db_loader.py
import os
import zipfile
import requests

def download_and_extract_databases(verbose=True):
    """
    Hugging Face에서 Vector DB ZIP 파일을 다운로드하고 압축을 푼다.
    """
    urls = {
        "chroma_db_law_real_final": "https://huggingface.co/datasets/sujeonggg/chroma_db_law_real_final/resolve/main/chroma_db_law_real_final.zip",
        "ja_chroma_db": "https://huggingface.co/datasets/sujeonggg/chroma_db_law_real_final/resolve/main/ja_chroma_db.zip",
    }

    def download_and_unzip(url, extract_to):
        os.makedirs(extract_to, exist_ok=True)
        zip_path = os.path.join(extract_to, "temp.zip")

        if os.path.exists(os.path.join(extract_to, "index")):
            if verbose:
                print(f"✅ Already exists: {extract_to}")
            return True

        try:
            if verbose:
                print(f"📦 Downloading from {url}...")
            r = requests.get(url)
            with open(zip_path, "wb") as f:
                f.write(r.content)

            if verbose:
                print(f"🧩 Unzipping to {extract_to}...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)

            os.remove(zip_path)
            return True
        except Exception as e:
            if verbose:
                print(f"❌ Failed to download {url}: {e}")
            return False

    success = True
    for name, url in urls.items():
        extract_path = name  # ZIP 이름 = 폴더명
        if not download_and_unzip(url, extract_path):
            success = False

    return success
