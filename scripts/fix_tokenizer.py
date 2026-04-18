"""
scripts/fix_tokenizer.py
-------------------------
Re-saves the tokenizer downloaded from Kaggle in a format
compatible with the local environment (no keras_preprocessing).

Run once from project root:
    python scripts/fix_tokenizer.py
"""

import pickle
import json
from pathlib import Path

TOKENIZER_PATH = Path("data/processed/tokenizer.pkl")
BACKUP_PATH    = Path("data/processed/tokenizer_kaggle_original.pkl")
JSON_PATH      = Path("data/processed/tokenizer.json")


def fix_tokenizer():
    print(f"Loading tokenizer from {TOKENIZER_PATH} ...")

    # Read raw bytes and unpickle carefully
    with open(TOKENIZER_PATH, "rb") as f:
        raw = f.read()

    # Patch the module reference before unpickling
    import io
    import copyreg

    patched = raw.replace(
        b"keras_preprocessing.text",
        b"tensorflow.keras.preprocessing.text"
    )

    tokenizer = pickle.loads(patched)
    print(f"Tokenizer loaded. Vocab size: {len(tokenizer.word_index) + 1}")

    # Back up the original
    BACKUP_PATH.write_bytes(raw)
    print(f"Original backed up to {BACKUP_PATH}")

    # Re-save with tensorflow.keras (local compatible)
    with open(TOKENIZER_PATH, "wb") as f:
        pickle.dump(tokenizer, f)
    print(f"Fixed tokenizer saved to {TOKENIZER_PATH}")

    # Also save as JSON (most portable format — no pickle dependency)
    tokenizer_json = tokenizer.to_json()
    JSON_PATH.write_text(tokenizer_json)
    print(f"Tokenizer also saved as JSON to {JSON_PATH}")

    print("\n✅ Done. Restart uvicorn now.")


if __name__ == "__main__":
    fix_tokenizer()