
import json
import pandas as pd
import argparse
import os

def save_tables_csv(json_file, output_base):
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        saved_files = []
        for i, table in enumerate(data):
            page_no = table.get('page', 'Unknown')
            records = table.get('data', [])
            
            if not records:
                print(f"Skipping empty table on page {page_no}")
                continue
                
            df = pd.DataFrame(records)
            
            # Construct filename
            # If only one table, use base name. If multiple, append page/index.
            if len(data) == 1:
                filename = f"{output_base}.csv"
            else:
                filename = f"{output_base}_p{page_no}_t{i+1}.csv"
                
            df.to_csv(filename, index=False, encoding='utf-8-sig') # utf-8-sig for Excel compatibility
            saved_files.append(filename)
            print(f"Saved table from page {page_no} to {filename}")
            
        return saved_files
            
    except Exception as e:
        print(f"Error converting to CSV: {e}")
        return []

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert JSON tables to CSV.")
    parser.add_argument("--input", type=str, required=True, help="Input JSON file path")
    parser.add_argument("--output", type=str, help="Output base filename (without extension)")
    
    args = parser.parse_args()
    
    input_path = args.input
    if args.output:
        output_base = args.output
    else:
        output_base = os.path.splitext(input_path)[0]
        
    save_tables_csv(input_path, output_base)
