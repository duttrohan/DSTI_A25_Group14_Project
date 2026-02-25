import os

PROJECT_ROOT = r"C:\Users\Comp\MY_PROJECT\Retail_Project"

DATA_RAW       = os.path.join(PROJECT_ROOT, "data", "raw")
DATA_INTERIM   = os.path.join(PROJECT_ROOT, "data", "interim")
DATA_PROCESSED = os.path.join(PROJECT_ROOT, "data", "processed")

os.makedirs(DATA_RAW, exist_ok=True)
os.makedirs(DATA_INTERIM, exist_ok=True)
os.makedirs(DATA_PROCESSED, exist_ok=True)
