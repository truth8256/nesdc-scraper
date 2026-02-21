import os
import shutil

# Patch symlink for Windows non-admin
def symlink_patch(src, dst, target_is_directory=False):
    if os.path.isdir(src):
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)

os.symlink = symlink_patch

# Setup Environment
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
os.environ["HF_HOME"] = r"d:\working\polldata\nesdc_scraper\hf_cache"

# Clear cache if needed (optional, but good to start fresh)
# if os.path.exists(os.environ["HF_HOME"]):
#     shutil.rmtree(os.environ["HF_HOME"], ignore_errors=True)

from docling.document_converter import DocumentConverter

def test_docling():
    # Target file
    target_file = r"d:\working\polldata\nesdc_scraper\data\raw\15380\3.통계표(260213)_MBC_2026년+설날특집+정치·사회현안+여론조사(2차)(15일보도).pdf"
    
    if not os.path.exists(target_file):
        print("Target file not found")
        return

    print(f"Converting {target_file}...")
    converter = DocumentConverter()
    result = converter.convert(target_file)
    markdown_output = result.document.export_to_markdown()
    
    print("\n--- Markdown Output ---\n")
    print(markdown_output)
    
    # Save to file for inspection
    with open("docling_test_15312.md", "w", encoding="utf-8") as f:
        f.write(markdown_output)
        
if __name__ == "__main__":
    test_docling()
