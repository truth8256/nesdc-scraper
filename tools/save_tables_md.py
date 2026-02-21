
import json
import pandas as pd
import sys

def save_tables_markdown(json_file, output_md):
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        with open(output_md, 'w', encoding='utf-8') as f_out:
            f_out.write(f"# Tables from {json_file}\n\n")
            
            for i, table in enumerate(data):
                page_no = table.get('page', 'Unknown')
                records = table.get('data', [])
                
                f_out.write(f"## Page {page_no} (Table {i+1})\n\n")
                
                if not records:
                    f_out.write("*Empty Table*\n\n")
                    continue
                    
                df = pd.DataFrame(records)
                f_out.write(df.to_markdown(index=False))
                f_out.write("\n\n")
                
        print(f"Saved tables to {output_md}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description="Convert JSON tables to Markdown.")
    parser.add_argument("--input", type=str, required=True, help="Input JSON file path")
    
    args = parser.parse_args()
    
    # Derivate output filename if not provided
    input_path = args.input
    output_path = input_path.replace(".json", "_view.md")
    
    save_tables_markdown(input_path, output_path)
