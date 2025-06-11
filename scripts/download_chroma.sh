# Google Drive에서 DB zip 파일 다운로드
curl -L -o .chroma/file1.zip "https://drive.google.com/uc?export=download&id=1gp5h0QScWB3wcsbs4i12ny1wEMY_HAqX"
curl -L -o .chroma/file2.zip "https://drive.google.com/uc?export=download&id=1dU9TLAPMg-Q8DLQjZM38CC-TsK477dSO"

# 압축 해제
unzip -o .chroma/file1.zip -d .chroma/
unzip -o .chroma/file2.zip -d .chroma/
