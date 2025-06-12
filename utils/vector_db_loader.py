import os
import zipfile
import requests

def download_and_extract_vector_db(url: str, extract_to: str) -> str:
    os.makedirs(extract_to, exist_ok=True)
    zip_path = os.path.join(extract_to, "temp.zip")

    # 이미 다운로드되어 있으면 스킵
    if os.path.exists(os.path.join(extract_to, "index")):
        print(f"✅ Already exists: {extract_to}")
        return extract_to

    print(f"📦 Downloading from {url}...")
    r = requests.get(url)
    with open(zip_path, "wb") as f:
        f.write(r.content)

    print(f"🧩 Unzipping to {extract_to}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

    os.remove(zip_path)
    return extract_to
