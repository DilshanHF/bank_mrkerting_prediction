"""
Download the Bank Marketing Dataset from UCI Repository.
Run this script first to get the dataset.
"""
import urllib.request
import zipfile
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip"
zip_path = os.path.join(DATA_DIR, "bank-additional.zip")

print("[1/3] Downloading dataset from UCI repository...")
urllib.request.urlretrieve(url, zip_path)
print(f"      Saved to {zip_path}")

print("[2/3] Extracting...")
with zipfile.ZipFile(zip_path, 'r') as z:
    z.extractall(DATA_DIR)
print(f"      Extracted to {DATA_DIR}")

# Move the CSV to a convenient location
import shutil
src = os.path.join(DATA_DIR, "bank-additional", "bank-additional-full.csv")
dst = os.path.join(DATA_DIR, "bank-additional-full.csv")
if os.path.exists(src) and not os.path.exists(dst):
    shutil.copy2(src, dst)

print("[3/3] Done! Dataset ready at:", dst)
print("      Rows: Use 'bank-additional-full.csv' (41,188 records, 20 features + target)")
