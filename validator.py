import json
import pandas as pd
import itertools
import re
import argparse
import os
import glob

def clean_number(value):
    """
    Cleans a string to extract a float value.
    Removes '%', '(', ')', ',' and handles '-' as 0.
    """
    if pd.isna(value):
        return 0.0
    if isinstance(value, float) or isinstance(value, int):
        return float(value)
    
    str_val = str(value).strip()
    
    if str_val in ['-', '.', '']:
        return 0.0
        
    # Remove characters
    # Keep decimal point
    str_val = re.sub(r'[%(),]', '', str_val)
    
    try:
        return float(str_val)
    except ValueError:
        return 0.0

def is_header_row(row_val):
    """Check if the value indicates a header row (e.g. '성별', '연령')"""
    keywords = ['성별', '연령', '지역', '권역', '직업', '학력', '소득', '이념', '지지정당', '기초', '광역']
    return any(k in str(row_val) for k in keywords)

def validate_table(df, table_info):
    """
    Validates a single dataframe.
    Returns a dict with validation results.
    """
    report = {
        'page': table_info.get('page'),
        'validity': 'Unknown',
        'issues': [],
        'score': 0
    }
    
    if df.empty:
        report['validity'] = 'Collection Impossible'
        report['issues'].append("Empty DataFrame")
        return report

    # 1. Column Identification
    base_cols = []
    header_cols = []
    value_cols = []
    
    for col in df.columns:
        col_str = str(col)
        # Simple heuristic for Base columns
        if any(x in col_str for x in ['사례수', 'Base', '가중값', '조사완료']):
            base_cols.append(col)
        # Heuristic for Header columns: usually the first 1-2 columns, or contain specific keywords if they are not the first
        # But for now let's assume the first column is always a header if it contains strings
        elif col == df.columns[0]: 
             header_cols.append(col)
        else:
            value_cols.append(col)
            
    if not base_cols:
        report['issues'].append("No Base/Case Count columns found")
        
    if not value_cols:
        report['issues'].append("No Value columns found")
        report['validity'] = 'Collection Impossible'
        return report

    # 2. Row Sum Validation
    valid_row_count = 0
    total_data_rows = 0
    
    for idx, row in df.iterrows():
        # Skip rows if the first column indicates it's a header repetition
        first_val = str(row[df.columns[0]])
        if is_header_row(first_val):
            continue
            
        # Extract values
        vals = [clean_number(row[c]) for c in value_cols]
        row_sum = sum(vals)
        
        # Check if row has any data (sum > 0)
        if row_sum > 0:
            total_data_rows += 1
            
            is_row_valid = False
            
            # Direct check
            if 98.0 <= row_sum <= 102.0:
                is_row_valid = True
            else:
                # Subset Sum Check (for Subtotals)
                # Filter positive values to reduce combinations
                pos_vals = [v for v in vals if v > 0]
                
                # If too many columns, skip subset check for performance
                if len(pos_vals) <= 20:
                    found_subset = False
                    # Check for any subset summing to 100
                    # We check combinations of length 1 to N
                    # Length 1 covers the case where "Total" column exists and is 100
                    for r in range(1, len(pos_vals) + 1):
                        for subset in itertools.combinations(pos_vals, r):
                            s = sum(subset)
                            if 98.0 <= s <= 102.0:
                                found_subset = True
                                is_row_valid = True
                                break
                        if found_subset:
                            break
            
            if is_row_valid:
                valid_row_count += 1
                
    row_validity_rate = 0
    if total_data_rows > 0:
        row_validity_rate = valid_row_count / total_data_rows * 100
        
    report['row_validity_rate'] = row_validity_rate
    
    # 3. Group Consistency (Simple Check)
    # We need to find "Totals" and "Subtotals" in the Base Column
    # This is hard without structured row headers.
    # For now, relying on Row Sums is a good proxy for "Is this a valid table?"
    
    # Classification
    if row_validity_rate >= 90:
        report['validity'] = 'Fully Valid'
    elif row_validity_rate >= 70:
        report['validity'] = 'Mostly Valid'
    else:
        report['validity'] = 'Collection Impossible'
        
    return report

def merge_split_tables(tables_data):
    """
    Attempts to merge tables that are split across page boundaries.
    Logic: If Table i and Table i+1 have the same Row Headers (first column), 
    append columns of Table i+1 to Table i.
    """
    merged_tables = []
    skip_indices = set()
    
    for i in range(len(tables_data)):
        if i in skip_indices:
            continue
            
        current_table = tables_data[i]
        merged_table = current_table.copy() # Start with current
        
        # Check next table
        if i + 1 < len(tables_data):
            next_table = tables_data[i+1]
            
            # Simple check: Compare first columns
            df1 = pd.DataFrame(current_table.get('data', []))
            df2 = pd.DataFrame(next_table.get('data', []))
            
            if not df1.empty and not df2.empty and len(df1) == len(df2):
                # Check similarity of first column
                col1 = df1.iloc[:, 0].astype(str).tolist()
                col2 = df2.iloc[:, 0].astype(str).tolist()
                
                if col1 == col2:
                    # Merge!
                    print(f"Merging tables from page {current_table.get('page')} and {next_table.get('page')}")
                    # Drop first column of df2 and concat
                    df2_vals = df2.iloc[:, 1:]
                    df_merged = pd.concat([df1, df2_vals], axis=1)
                    
                    merged_table['data'] = df_merged.to_dict(orient='records')
                    merged_table['merged_pages'] = [current_table.get('page'), next_table.get('page')]
                    
                    merged_tables.append(merged_table)
                    skip_indices.add(i+1)
                    continue

        merged_tables.append(merged_table)
        
    return merged_tables

def run_validation(json_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    # Pre-merging step
    merged_data = merge_split_tables(data)
    
    results = []
    for table_info in merged_data:
        df = pd.DataFrame(table_info.get('data', []))
        report = validate_table(df, table_info)
        results.append(report)
        
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input JSON file")
    args = parser.parse_args()
    
    reports = run_validation(args.input)
    
    print(f"\nValidation Report for {args.input}")
    print("="*60)
    for r in reports:
        pages = r.get('page')
        if 'merged_pages' in r: # From manual merge logic if we passed it in report
           pages = f"{r.get('merged_pages')}"
           
        print(f"Page: {pages} | Status: {r['validity']} | Row Validity: {r.get('row_validity_rate', 0):.1f}%")
        if r['issues']:
            for issue in r['issues']:
                print(f"  - {issue}")
    print("="*60)
