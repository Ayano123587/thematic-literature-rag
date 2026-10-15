from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MODEL = ROOT / "models" / "index.pkl"
BOOKS = {
    "pride": {"id": 1342, "title": "Pride and Prejudice", "author": "Jane Austen", "death": 1817},
    "frankenstein": {"id": 84, "title": "Frankenstein", "author": "Mary Shelley", "death": 1851},
}
