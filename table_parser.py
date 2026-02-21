import os
import json
import shutil
import pandas as pd
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
# Use default HF cache (don't override HF_HOME)
# This allows Docling to find its downloaded models

from docling.document_converter import DocumentConverter
from validator import validate_table
from llm_extractor import extract_table_from_markdown
from schema import convert_flat_to_standard, validate_schema



def extract_question_from_markdown(md_text):
    """
    Markdown 텍스트에서 질문 추출.

    Parameters
    ----------
    md_text : str
        Docling이 변환한 markdown 텍스트

    Returns
    -------
    str  질문 텍스트 (없으면 빈 문자열)
    """
    import re

    # "Q숫자.", "SQ숫자.", "문 숫자." 등으로 시작하는 줄 찾기
    lines = md_text.split('\n')
    question_pattern = re.compile(r'^(Q\d+\.|SQ\d+\.|문\s*\d+\.)')

    for i, line in enumerate(lines):
        line = line.strip()
        if question_pattern.match(line):
            # 여러 줄에 걸쳐 있을 수 있으므로 다음 줄도 포함
            question = line
            for j in range(i + 1, min(i + 5, len(lines))):
                next_line = lines[j].strip()
                # 새로운 질문이나 테이블 시작 패턴이 나오면 중단
                if question_pattern.match(next_line) or next_line.startswith('|') or next_line.startswith('#'):
                    break
                if next_line:
                    question += " " + next_line
            return question.strip()

    return ""


def get_target_pdf(dir_path):
    """
    Identify the target PDF in the directory.
    Strategy: Select the LARGEST PDF file (likely the Result Report).
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
        
    # Sort by size (descending) - LARGEST first
    file_info.sort(key=lambda x: x[1], reverse=True)
    

    
    return file_info[0][2] # Return full path of largest file

def split_merged_columns(df):
    """
    Heuristic to split columns that have been merged by Docling.
    Logic: If a column header has N space-separated tokens, and the values in that column
    also have N space-separated tokens (consistent across most rows), split them.
    """
    if df.empty:
        return df
        
    # We will build a new DataFrame column by column
    # To handle duplicate column names in input, we iterate by integer index
    new_cols = {}
    new_col_names = []
    
    # helper to generate unique name
    def get_unique_name(base_name, existing_names):
        name = base_name
        counter = 1
        while name in existing_names:
            name = f"{base_name}_{counter}"
            counter += 1
        return name

    for i in range(len(df.columns)):
        col_name = df.columns[i]
        col_data = df.iloc[:, i]
        
        col_str = str(col_name).strip()
        header_tokens = col_str.split()
        
        should_split = False
        
        # Candidate for splitting?
        if len(header_tokens) > 1:
            # Check consistency
            non_empty_rows = col_data.dropna().astype(str)
            non_empty_rows = non_empty_rows[non_empty_rows.str.strip() != ""]
            
            if len(non_empty_rows) > 0:
                match_count = 0
                mismatch = False
                
                for val in non_empty_rows:
                    tokens = val.strip().split()
                    if len(tokens) == len(header_tokens):
                        match_count += 1
                    elif len(tokens) > len(header_tokens):
                        mismatch = True 
                        break
                        
                if not mismatch and (match_count / len(non_empty_rows) > 0.5):
                    should_split = True
        
        if should_split:
            print(f"  [Auto-Split] Splitting column '{col_str}' into {len(header_tokens)} columns.")
            
            split_data = {t: [] for t in header_tokens}
            
            for val in col_data:
                val_str = str(val).strip()
                if not val_str:
                    for t in header_tokens:
                        split_data[t].append("")
                    continue
                    
                tokens = val_str.split()
                if len(tokens) == len(header_tokens):
                    for idx, t in enumerate(header_tokens):
                        split_data[t].append(tokens[idx])
                else:
                    # Fallback
                    split_data[header_tokens[0]].append(val_str)
                    for idx in range(1, len(header_tokens)):
                        split_data[header_tokens[idx]].append("")
            
            # Add split columns to new_cols
            for t in header_tokens:
                unique_name = get_unique_name(t, new_col_names)
                new_col_names.append(unique_name)
                new_cols[unique_name] = split_data[t]
        else:
            # Keep original
            unique_name = get_unique_name(str(col_name), new_col_names)
            new_col_names.append(unique_name)
            new_cols[unique_name] = col_data.values
            
    return pd.DataFrame(new_cols)

def parse_survey_table(folder_name, page_numbers, force_llm=False):
    """
    Extract tables from specific pages of the largest PDF in the folder.
    Optimized to convert only the requested pages.
    """
    # 현재 스크립트 디렉토리 기준 상대 경로
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(script_dir, "data", "raw")
    dir_path = os.path.join(base_dir, str(folder_name))
    
    target_pdf = get_target_pdf(dir_path)
    if not target_pdf:
        print(f"No PDF found in {dir_path}")
        return
        
    print(f"Targeting file: {target_pdf}")
    
    try:
        converter = DocumentConverter()
        print("Model initialized. Starting conversion...")
        
        extracted_tables = []
        
        # Sort and deduplicate pages
        pages_sorted = sorted(list(set(page_numbers)))
        
        for p in pages_sorted:
            print(f"Processing page {p}...")
            # Use page_range=(p, p) to convert only this page
            try:
                result = converter.convert(target_pdf, page_range=(p, p))
                doc = result.document

                # --- Extract question from markdown ---
                md_text = doc.export_to_markdown()
                question_text = extract_question_from_markdown(md_text)
                if question_text:
                    print(f"  - Question: {question_text[:80]}...")

                # --- 1. Docling Extraction & Validation ---
                docling_candidates = []

                for table in doc.tables:
                    # Validate provenance if available
                    if table.prov:
                        page_no_in_doc = table.prov[0].page_no
                    else:
                        page_no_in_doc = p # Fallback
                    
                    # Export to DataFrame
                    df = table.export_to_dataframe()
                    
                    # --- Post-Process: Split Merged Columns ---
                    df = split_merged_columns(df)
                    
                    # --- Validation Check ---
                    # Check if this table is "good enough"
                    report = validate_table(df, {'page': p})
                    score = report.get('row_validity_rate', 0)
                    status = report.get('validity', 'Unknown')
                    
                    print(f"  - Found table on page {p}. Validity: {status} ({score:.1f}%)")
                    
                    docling_candidates.append({
                        "page": page_no_in_doc, 
                        "data": df.to_dict(orient="records"),
                        "columns": df.columns.tolist(),
                        "score": score,
                        "status": status,
                        "df": df 
                    })

                # --- 2. Fallback Decision ---
                use_llm = False
                
                if force_llm:
                    print(f"  - [Force LLM] Triggering LLM extraction as requested.")
                    use_llm = True
                elif not docling_candidates:
                    print(f"  - No tables found by Docling on page {p}. Triggering LLM Fallback...")
                    use_llm = True
                else:
                    # Check if best candidate is acceptable
                    # If best score is < 15% (Collection Impossible), trigger LLM
                    best_score = max(c['score'] for c in docling_candidates)
                    if best_score < 15.0:
                        print(f"  - Docling tables have low validity (Max: {best_score:.1f}%). Triggering LLM Fallback...")
                        use_llm = True
                
                # --- 3. Execution ---
                if use_llm:
                    print(f"  - Invoking Gemini LLM on page {p} markdown...")
                    md_text = doc.export_to_markdown()
                    context_msg = f"Survey data from page {p}. Found columns might be merged."
                    
                    llm_result = extract_table_from_markdown(md_text, context=context_msg)
                    
                    llm_data_list = []
                    question_text = ""
                    
                    if isinstance(llm_result, dict) and 'data' in llm_result:
                        llm_data_list = llm_result['data']
                        question_text = llm_result.get('question', '')
                    elif isinstance(llm_result, list):
                        llm_data_list = llm_result
                    
                    if llm_data_list:
                         # Prepare for validation: Flatten if necessary
                         flat_data = []
                         for row in llm_data_list:
                             if 'responses' in row and isinstance(row['responses'], dict):
                                 # Create a flat copy
                                 new_row = row.copy()
                                 responses = new_row.pop('responses')
                                 for k, v in responses.items():
                                     new_row[k] = v
                                 flat_data.append(new_row)
                             else:
                                 flat_data.append(row)

                         # Validation DataFrame
                         val_df = pd.DataFrame(flat_data)

                         # Basic cleanup for LLM output (ensure consistency)
                         llm_report = validate_table(val_df, {'page': p})
                         print(f"  - LLM Extraction Result: {llm_report['validity']} ({llm_report.get('row_validity_rate', 0):.1f}%)")

                         # Convert to standard schema
                         # LLM already has nested structure, so convert from flat
                         standard_table = convert_flat_to_standard(
                             flat_data,
                             poll_id=str(folder_name),
                             page=p,
                             keyword="",
                             question=question_text if question_text else llm_result.get('question', ''),
                             method="llm_gemini_v1.0"
                         )

                         # Schema validation
                         schema_issues = validate_schema(standard_table)
                         if schema_issues:
                             print(f"  - Schema validation warnings:")
                             for issue in schema_issues[:3]:
                                 print(f"    • {issue}")

                         extracted_tables.append(standard_table)
                    else:
                         print("  - LLM failed to extract meaningful data.")
                         # Fallback to Docling result if it existed, just to save something
                         if docling_candidates:
                             print("  - Reverting to low-quality Docling output.")
                             for c in docling_candidates:
                                 df = c['df']
                                 flat_data = df.to_dict(orient='records')

                                 # Convert to standard schema
                                 standard_table = convert_flat_to_standard(
                                     flat_data,
                                     poll_id=str(folder_name),
                                     page=p,
                                     keyword="",
                                     question=question_text,
                                     method="docling_v1.0_fallback"
                                 )

                                 extracted_tables.append(standard_table)

                else:
                    # Use Docling results - Convert to standard schema
                    for c in docling_candidates:
                        df = c['df']
                        flat_data = df.to_dict(orient='records')

                        # Convert to standard schema
                        standard_table = convert_flat_to_standard(
                            flat_data,
                            poll_id=str(folder_name),
                            page=p,
                            keyword="",  # 자동 추론은 Phase 3에서 구현
                            question=question_text,
                            method="docling_v1.0"
                        )

                        # Schema validation
                        schema_issues = validate_schema(standard_table)
                        if schema_issues:
                            print(f"  - Schema validation warnings:")
                            for issue in schema_issues[:3]:
                                print(f"    • {issue}")

                        extracted_tables.append(standard_table)
                    
            except Exception as e:
                print(f"  - Error processing page {p}: {e}")
                import traceback
                traceback.print_exc()

        # Save output
        output_dir = os.path.join(script_dir, "data", "parsed_tables")
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"{folder_name}_docling.json")
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(extracted_tables, f, ensure_ascii=False, indent=2)

        print(f"\n{'='*60}")
        print(f"Saved {len(extracted_tables)} tables to {output_file}")
        print(f"{'='*60}")
        
    except Exception as e:
        print(f"Error processing {target_pdf}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract tables from PDF survey results.")
    parser.add_argument("--folder", type=str, required=True, help="Folder name (e.g., 15312)")
    parser.add_argument("--pages", type=int, nargs="+", required=True, help="List of page numbers to extract (e.g., 25 27 41)")
    parser.add_argument("--force-llm", action="store_true", help="Force LLM extraction regardless of validation score")
    
    args = parser.parse_args()
    
    print(f"Processing Folder: {args.folder}, Pages: {args.pages}, Force LLM: {args.force_llm}")
    parse_survey_table(args.folder, args.pages, force_llm=args.force_llm)
