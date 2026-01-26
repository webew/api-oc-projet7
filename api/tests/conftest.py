import sys
from pathlib import Path

API_DIR = Path(__file__).resolve().parents[1]  # .../api
sys.path.insert(0, str(API_DIR))
