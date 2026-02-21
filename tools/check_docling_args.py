
import inspect
import sys
import os

# --- Environment Setup for Docling/HF ---
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
cache_dir = r"d:\working\polldata\nesdc_scraper\hf_cache"
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
os.environ["HF_HOME"] = cache_dir

from docling.document_converter import DocumentConverter

print("Arguments for DocumentConverter.convert:")
sig = inspect.signature(DocumentConverter.convert)
print(sig)
