
import json
import pandas as pd
import sys

def view_tables(json_file):
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        for i, table in enumerate(data):
            page_no = table.get('page', 'Unknown')
            records = table.get('data', [])
            
            if not records:
                print(f"\n--- Page {page_no} (Empty Table) ---")
                continue
                
            df = pd.DataFrame(records)
            print(f"\n--- Page {page_no} (Table {i+1}) ---")
            print(df.to_markdown(index=False)) # Use markdown format for nice table
            print("\n")
            
    except Exception as e:
        print(f"Error reading {json_file}: {e}")

if __name__ == "__main__":
    target_file = r"d:\working\polldata\nesdc_scraper\data\parsed_tables\15312_tables.json"
    view_tables(target_file)
