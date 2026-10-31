import json
import requests
import urllib.request
import os
from glob import glob
import pandas as pd
from typing import List, Union

BASE_URL = 'https://api.figshare.com/v2'

def downloaddata(item_id: Union[int, str], folder: str, base_url: str = BASE_URL) -> None:
    """Downloads files from a Figshare article."""
    item_id = str(item_id)

    if not isinstance(folder, str):
        raise ValueError(f"folder must be a string, got {type(folder).__name__}")

    os.makedirs(folder, exist_ok=True)

    try:
        response = requests.get(f"{base_url}/articles/{item_id}")
        response.raise_for_status()
        metadata = response.json()
        doi = metadata.get("doi")
        print(f"\033[1m\033[93mIf using this data please cite https://doi.org/{doi}\033[0m")
    except requests.RequestException as e:
        print(f"[ERROR] Failed to fetch metadata for ITEM_ID={item_id}: {e}")
        return
    except json.JSONDecodeError:
        print(f"[ERROR] Response is not valid JSON for ITEM_ID={item_id}")
        return

    files = metadata.get('files', [])
    if not files:
        print(f"[INFO] No files found in ITEM_ID={item_id}")
        return

    for file in files:
        name = file.get('name')
        url = file.get('download_url')
        if not name or not url:
            print(f"[WARNING] Skipping file due to missing name or URL: {file}")
            continue

        file_path = os.path.join(folder, name)
        try:
            print(f"[INFO] Downloading {name} -> {file_path}")
            urllib.request.urlretrieve(url, file_path)
        except Exception as e:
            print(f"[ERROR] Failed to download {name}: {e}")


def mergefiles(src_dir: str, dest_dir: str) -> None:
    """Merges CSVs with the same name across subfolders into single files."""
    if not os.path.isdir(src_dir):
        print(f"[ERROR] Source directory does not exist: {src_dir}")
        return

    os.makedirs(dest_dir, exist_ok=True)
    folders = [f for f in glob(os.path.join(src_dir, '*')) if os.path.isdir(f)]
    if not folders:
        print(f"[INFO] No subdirectories found in {src_dir}.")
        return

    sample_folder = folders[0]
    base_files = glob(os.path.join(sample_folder, '*.csv'))
    if not base_files:
        print(f"[INFO] No CSV files found in: {sample_folder}")
        return

    for base_file in base_files:
        filename = os.path.basename(base_file)
        try:
            combined_df = pd.read_csv(base_file)
        except Exception as e:
            print(f"[ERROR] Failed to read {base_file}: {e}")
            continue

        for folder in folders[1:]:
            other_file = os.path.join(folder, filename)
            if os.path.exists(other_file):
                try:
                    df_temp = pd.read_csv(other_file)
                    combined_df = pd.concat([combined_df, df_temp], ignore_index=True)
                except Exception as e:
                    print(f"[ERROR] Could not read {other_file}: {e}")
            else:
                print(f"[WARNING] {filename} missing in {folder}, skipping.")

        dest_path = os.path.join(dest_dir, filename)
        try:
            combined_df.to_csv(dest_path, index=False)
            print(f"[SUCCESS] Merged: {dest_path}")
        except Exception as e:
            print(f"[ERROR] Could not save merged file {dest_path}: {e}")


def getdata(itemids: List[Union[int, str]], download_dir: str, datadir: str = 'data', base_url: str = BASE_URL) -> None:
    """Downloads and merges Figshare articles by ID."""
    if not isinstance(itemids, list) or not all(isinstance(i, (int, str)) for i in itemids):
        raise ValueError("itemids must be a list of strings or integers.")

    os.makedirs(download_dir, exist_ok=True)

    for item_id in itemids:
        item_str = str(item_id)
        folder = os.path.join(download_dir, item_str)
        os.makedirs(folder, exist_ok=True)
        downloaddata(item_str, folder, base_url=base_url)

    mergefiles(download_dir, datadir)
