import pdfplumber
import os

target_dirs = ["15312", "15311", "13808"]
base_path = r"d:\working\polldata\nesdc_scraper\data\raw"

print("Starting analysis...")
with open("pdf_analysis.txt", "w", encoding="utf-8") as f:
    for ntt_id in target_dirs:
        dir_path = os.path.join(base_path, ntt_id)
        if not os.path.exists(dir_path):
            f.write(f"Directory not found: {dir_path}\n")
            continue
            
        all_files = os.listdir(dir_path)
        f.write(f"\n--- Analyzing {ntt_id} ---\n")
        f.write(f"All files: {all_files}\n")

        pdf_files = [fn for fn in all_files if fn.lower().endswith('.pdf')]
        
        if not pdf_files:
            f.write(f"No PDF files in {ntt_id}\n")
            continue
            
        # Sort by size
        files_with_size = []
        for fn in pdf_files:
            full_path = os.path.join(dir_path, fn)
            size = os.path.getsize(full_path)
            files_with_size.append((fn, size, full_path))
            
        files_with_size.sort(key=lambda x: x[1]) # Smallest first
            
        target_file = files_with_size[0] # Smallest file
        f.write(f"Target (Smallest): {target_file[0]} ({target_file[1]} bytes)\n")
        
        try:
            with pdfplumber.open(target_file[2]) as pdf:
                f.write(f"Total Pages: {len(pdf.pages)}\n")
                # Print first 2 pages text
                for i, page in enumerate(pdf.pages):
                    if i >= 3: break # First 3 pages
                    text = page.extract_text()
                    f.write(f"\n[Page {i+1}]\n{text}\n") 
                    f.write("-" * 40 + "\n")
        except Exception as e:
            f.write(f"Error reading PDF: {e}\n")
