"""
Demographic Model Weights Downloader
Downloads pre-trained DeepFace weights for:
- Age Estimation (age_model_weights.h5)
- Gender Classification (gender_model_weights.h5)
- Demographic/Race Classification (race_model_single_batch.h5)
- Emotion Recognition (facial_expression_model_weights.h5)

Includes:
- SSL certificate verification via certifi
- Resumable streaming downloads
- Progress reporting
- Windows UTF-8 compatibility
"""

import os
import sys
import time

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import requests
import certifi

WEIGHTS_DIR = os.path.join(os.path.expanduser("~"), ".deepface", "weights")

MODELS = {
    "facial_expression_model_weights.h5": {
        "url": "https://github.com/serengil/deepface_models/releases/download/v1.0/facial_expression_model_weights.h5",
        "description": "Facial Emotion Recognition Model (~6 MB)"
    },
    "age_model_weights.h5": {
        "url": "https://github.com/serengil/deepface_models/releases/download/v1.0/age_model_weights.h5",
        "description": "Apparent Age Estimation Model (~513 MB)"
    },
    "gender_model_weights.h5": {
        "url": "https://github.com/serengil/deepface_models/releases/download/v1.0/gender_model_weights.h5",
        "description": "Gender Classification Model (~512 MB)"
    },
    "race_model_single_batch.h5": {
        "url": "https://github.com/serengil/deepface_models/releases/download/v1.0/race_model_single_batch.h5",
        "description": "Broad Demographic/Ethnic Proxy Model (~512 MB)"
    }
}


def download_file(url, dest_path, description=""):
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    temp_path = dest_path + ".part"

    resume_header = {}
    downloaded_bytes = 0
    if os.path.exists(temp_path):
        downloaded_bytes = os.path.getsize(temp_path)
        resume_header = {"Range": f"bytes={downloaded_bytes}-"}

    print(f"\n[DOWNLOAD] {description}")
    print(f"[TARGET] {dest_path}")

    try:
        response = requests.get(
            url,
            headers=resume_header,
            stream=True,
            verify=certifi.where(),
            timeout=30
        )
    except Exception as e:
        print(f"[ERROR] Failed to connect to server: {e}")
        return False

    total_bytes = None
    if "Content-Range" in response.headers:
        # Format: bytes start-end/total
        total_bytes = int(response.headers["Content-Range"].split("/")[-1])
    elif "Content-Length" in response.headers:
        total_bytes = int(response.headers["Content-Length"]) + downloaded_bytes

    mode = "ab" if downloaded_bytes > 0 and response.status_code == 206 else "wb"
    if mode == "wb":
        downloaded_bytes = 0

    chunk_size = 1024 * 1024  # 1MB chunks
    start_time = time.time()
    last_print = 0

    with open(temp_path, mode) as f:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                downloaded_bytes += len(chunk)
                curr_time = time.time()
                if curr_time - last_print > 0.5:
                    elapsed = max(curr_time - start_time, 0.001)
                    speed_mb = (downloaded_bytes / (1024 * 1024)) / elapsed
                    if total_bytes:
                        pct = (downloaded_bytes / total_bytes) * 100.0
                        total_mb = total_bytes / (1024 * 1024)
                        dl_mb = downloaded_bytes / (1024 * 1024)
                        print(f"\r  -> Progress: {dl_mb:.1f} MB / {total_mb:.1f} MB ({pct:.1f}%) | Speed: {speed_mb:.2f} MB/s", end="", flush=True)
                    else:
                        print(f"\r  -> Downloaded: {downloaded_bytes / (1024*1024):.1f} MB | Speed: {speed_mb:.2f} MB/s", end="", flush=True)
                    last_print = curr_time

    print()
    # Rename finished file
    if os.path.exists(dest_path):
        os.remove(dest_path)
    os.rename(temp_path, dest_path)
    print(f"[SUCCESS] Finished downloading: {os.path.basename(dest_path)}")
    return True


def check_and_download_all():
    print("=" * 70)
    print("DeepFace Demographic & Emotion Pre-trained Model Weights Manager")
    print(f"Target Directory: {WEIGHTS_DIR}")
    print("=" * 70)

    for filename, info in MODELS.items():
        dest = os.path.join(WEIGHTS_DIR, filename)
        if os.path.exists(dest) and os.path.getsize(dest) > 1000000:
            size_mb = os.path.getsize(dest) / (1024 * 1024)
            print(f"[FOUND] {filename} is already downloaded ({size_mb:.1f} MB). Skipping.")
        else:
            download_file(info["url"], dest, info["description"])

    print("\n" + "=" * 70)
    print("All demographic weights are configured and ready for real-time inference!")
    print("=" * 70)


if __name__ == "__main__":
    check_and_download_all()
