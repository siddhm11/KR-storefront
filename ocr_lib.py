import requests
import json
import pandas as pd
import io
import streamlit as st

OCR_API_URL = 'https://api.ocr.space/parse/image'

def get_carrefour_data(pdf_file, api_key='helloworld'):
    """
    Sends PDF to OCR.space API and parses the result into a DataFrame.
    """
    # Reset file pointer
    pdf_file.seek(0)
    
    payload = {
        'apikey': api_key,
        'isOverlayRequired': True,
        'detectOrientation': True, # Important for "sideways" faxes
        'scale': True,
        'OCREngine': 2,
        'isTable': True,
    }
    
    files = {'file': pdf_file}
    
    try:
        response = requests.post(OCR_API_URL, files=files, data=payload, timeout=30)
        result = response.json()
        
        if result.get('IsErroredOnProcessing') or result.get('OCRExitCode') == 3:
            err = result.get('ErrorMessage')
            st.error(f"OCR API Error: {err}")
            return None
            
        return parse_ocr_json(result)
        
    except Exception as e:
        st.error(f"OCR Request Failed: {e}")
        return None

def parse_ocr_json(data):
    """
    Parses OCR.space JSON output for Carrefour attributes.
    Logic: Rows are grouped by 'Left' coordinate (approx).
           Columns are identified by 'Top' coordinate.
    """
    if 'ParsedResults' not in data:
        return None
        
    # Combine lines from all pages? unique logic might be needed per page if layout differs
    # But usually Carrefour fax is consistent.
    
    rows_data = []
    
    for page in data['ParsedResults']:
        if 'TextOverlay' not in page:
            continue
            
        lines = page['TextOverlay']['Lines']
        
        # 1. Group by Left coordinate (Rows)
        # We need to cluster lines that have similar Left values.
        # Simple clustering: Sort by Left, then group.
        
        # Helper to get line Left
        def get_left(line):
            return line['Words'][0]['Left']
            
        sorted_lines = sorted(lines, key=get_left, reverse=True) # Descending Left (Top-Bottom presumably)
        
        current_group = []
        current_left = -1
        TOLERANCE = 20.0
        
        groups = []
        
        for line in sorted_lines:
            l_val = get_left(line)
            if current_left == -1:
                current_left = l_val
                current_group.append(line)
            elif abs(l_val - current_left) < TOLERANCE:
                current_group.append(line)
            else:
                groups.append(current_group)
                current_group = [line]
                current_left = l_val
        if current_group:
            groups.append(current_group)
            
        # 2. Parse columns in each group
        for group in groups:
            row_dict = {}
            # Columns by Top range
            # Barcode: ~120-150
            # Description: ~1000-1600 (could be multiple lines)
            # Quantity: ~2000-2100
            
            desc_parts = []
            
            for line in group:
                top = line['Words'][0]['Top']
                text = line['LineText']
                
                # Barcode Logic
                if 100 <= top <= 400:
                    # Clean barcode (remove spaces)
                    bc = text.replace(" ", "").strip()
                    if bc.isdigit() and len(bc) > 8:
                        row_dict['Barcode'] = bc
                        
                # Description Logic
                elif 900 <= top <= 1800:
                    desc_parts.append(text)
                    
                # Quantity Logic
                elif 2000 <= top <= 2200:
                    # Quantity line usually contains just digits or "10"
                    # But sometimes noise. 
                    # Try to parse int
                    q_text = text.replace("|", "").strip() # frequent OCR artifact
                    if q_text.isdigit():
                        row_dict['Quantity'] = int(q_text)
                    else:
                        # try lenient parse
                        import re
                        nums = re.findall(r'\d+', q_text)
                        if nums: row_dict['Quantity'] = int(nums[0])

            if desc_parts:
                row_dict['Description'] = " ".join(desc_parts)
                
            # Filter valid rows: Must have Barcode and Quantity (or at least Barcode)
            if 'Barcode' in row_dict:
                rows_data.append(row_dict)

    if not rows_data:
        return None
        
    df = pd.DataFrame(rows_data)
    # Add generic UOM as 'EA' if missing, or handle logic
    df['UOM'] = 'EA' 
    return df
