import os
import json
import re
import shutil

# --- Patch for Symlinks on Windows (Non-Admin) ---
def symlink_patch(src, dst, target_is_directory=False):
    if os.path.isdir(src):
        if os.path.exists(dst):
            shutil.rmtree(dst)
        try:
            shutil.copytree(src, dst)
        except Exception:
            pass
    else:
        try:
            shutil.copy2(src, dst)
        except Exception:
            pass

os.symlink = symlink_patch

# --- Environment Setup for Docling/HF ---
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
os.environ["HF_HOME"] = r"d:\working\polldata\nesdc_scraper\hf_cache"

from docling.document_converter import DocumentConverter

def get_target_pdf(dir_path):
    """
    Identify the target PDF in the directory.
    Strategy: Select the SMALLEST PDF file.
    """
    if not os.path.exists(dir_path):
        return None
        
    files = [f for f in os.listdir(dir_path) if f.lower().endswith(".pdf")]
    
    if not files:
        return None
        
    # Get full paths and sizes
    file_info = []
    for f in files:
        full_path = os.path.join(dir_path, f)
        size = os.path.getsize(full_path)
        file_info.append((f, size, full_path))
        
    # Sort by size (ascending) - smallest first
    file_info.sort(key=lambda x: x[1])
    
    return file_info[0][2] # Return full path of smallest file

def parse_markdown_survey(text):
    """
    Parse the Markdown text to extract questions and options.
    """
    # 1. Cleanup escaped characters from Docling markdown
    text = text.replace(r"\_", "_")
    
    # 2. Find Question Starts
    # Patterns: SQ1, Q1, 문1, DQ1
    # Markdown might wrap them in bold **Q1.** or be inside table | Q1. |
    # Regex should handle optional table delimiters and whitespace/asterisks
    # Example: | SQ3. ...
    # Example: **Q1.** ...
    qr_pattern = r'(?:^|[\s|*])(SQ\s*\d+|Q\s*\d+(?:[-_]\d+)?|문\s*\d+|DQ\s*\d+)\.?\s*'
    
    # Use finditer to locate all questions
    matches = list(re.finditer(qr_pattern, text, re.IGNORECASE))
    
    parsed_data = []
    
    for i in range(len(matches)):
        match = matches[i]
        start_idx = match.start()
        
        # Determine end of this question section (start of next question or end of text)
        if i + 1 < len(matches):
            end_idx = matches[i+1].start()
        else:
            end_idx = len(text)
            
        full_segment = text[start_idx:end_idx]
        
        # Extract ID
        q_id = match.group(1).replace(" ", "")
        
        # Content content follows the match end
        content_extracted = text[match.end():end_idx]
        
        # Clean up Markdown artifacts from content
        content_clean = content_extracted.strip()
        
        # Remove extraction artifacts like " | " at the beginning/end
        content_clean = re.sub(r'^\|\s*', '', content_clean)
        content_clean = re.sub(r'\s*\|$', '', content_clean)
        # Remove ** if present
        content_clean = content_clean.replace("**", "")
        
        # 3. Extract Options
        # Patterns: 1번, (1), ①, 1.
        # Often in Docling MD, options might be like "1 번 ," or "| 1 번 ,"
        opt_pattern_str = r'(?:^|\s)(?:\|?\s*)(\d+\s*번|\(\d+\)|[①-⑮]|\d+\.)'
        opt_matcher = re.compile(opt_pattern_str)
        
        opt_matches = list(opt_matcher.finditer(content_clean))
        
        final_text = content_clean
        options = []
        
        if opt_matches:
            first_opt_match = opt_matches[0]
            # Text is everything before the first option
            # But we must act relative to content_clean
            first_opt_start = first_opt_match.start()
            final_text = content_clean[:first_opt_start].strip()
            
            for k in range(len(opt_matches)):
                o_match = opt_matches[k]
                o_start = o_match.start()
                if k + 1 < len(opt_matches):
                    o_end = opt_matches[k+1].start()
                else:
                    o_end = len(content_clean)
                
                opt_segment = content_clean[o_start:o_end]
                
                # Label is in group 1
                label = o_match.group(1)
                
                # Remove label from segment to get value
                # Identify where label starts in segment (it should be near start)
                # We can essentially just strip the label string
                # Be careful if label appears twice
                
                # Simple approach: Replace first occurrence
                val = opt_segment.replace(label, "", 1).strip()
                
                # Clean value (remove trailing punctuation commonly found)
                val = re.sub(r'[,|]+$', '', val).strip()
                
                options.append(f"{label} {val}")
        
        # Final cleanup of text
        final_text = final_text.replace("\n", " ").strip()
        final_text = re.sub(r'\s+', ' ', final_text)
        
        parsed_data.append({
            "id": q_id,
            "text": final_text,
            "options": options,
            "footnote": "" 
        })
        
    return parsed_data

def extract_from_poll(ntt_id, base_path):
    dir_path = os.path.join(base_path, ntt_id)
    target_pdf = get_target_pdf(dir_path)
    
    if not target_pdf:
        # print(f"[{ntt_id}] No PDF found.")
        return None
        
    print(f"[{ntt_id}] Processing {os.path.basename(target_pdf)}...")
    
    try:
        # Convert with Docling
        converter = DocumentConverter()
        result = converter.convert(target_pdf)
        md_text = result.document.export_to_markdown()
        
        # Parse
        questions = parse_markdown_survey(md_text)
        
        print(f"[{ntt_id}] Extracted {len(questions)} questions.")
        return {"nttId": ntt_id, "questions": questions}
        
    except Exception as e:
        print(f"[{ntt_id}] Error processing PDF: {e}")
        return None

if __name__ == "__main__":
    # Target all directories in data/raw
    base_dir = r"d:\working\polldata\nesdc_scraper\data\raw"
    output_file = os.path.join(base_dir, "survey_data_all.json")
    
    if os.path.exists(base_dir):
        subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
        
        # Process recent ones first
        subdirs.sort(reverse=True)
        
        # Limit to top 10 for testing
        top_10 = subdirs[:10]
        
        all_data = []
        count = 0
        
        print(f"Processing top {len(top_10)} directories...")
        
        for nid in top_10:
            result = extract_from_poll(nid, base_dir)
            if result:
                all_data.append(result)
                count += 1
        
        # Save aggregated file
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(all_data, f, ensure_ascii=False, indent=2)
            
        print(f"Total processed: {count}")
        print(f"Saved aggregated data to {output_file}")
    else:
        print(f"Base directory not found: {base_dir}")
