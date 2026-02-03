from pathlib import Path

# Definer projektets rod 
ROOT_DIR = Path(__file__).resolve().parent.parent

# Definer centrale mapper
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_DIR = ROOT_DIR / "models"
TRAINED_MODEL_DIR = MODEL_DIR / "trained"

# Opret mapperne hvis de ikke eksisterer
for path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODEL_DIR, TRAINED_MODEL_DIR]:
    path.mkdir(parents=True, exist_ok=True)