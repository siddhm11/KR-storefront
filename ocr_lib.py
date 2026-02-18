"""
OCR Library for Carrefour Fax Processing.

Supports two OCR engines:
1. ocrwebservice.com (Primary) — Same engine as onlineocr.net, outputs Excel directly.
   Free trial: 25 pages/day. Sign up at https://www.ocrwebservice.com/account/signup
2. OCR.space (Fallback) — Free but lower quality for tables.

Also handles parsing of manually-converted Excel files from onlineocr.net.
"""
import requests
import json
import pandas as pd
import io
import re
import streamlit as st

# ============================================================
# ENGINE 1: ocrwebservice.com (Primary — Excel output)
# ============================================================
OCR_WEB_SERVICE_URL = 'http://www.ocrwebservice.com/restservices/processDocument'

def ocr_via_webservice(pdf_file, username, license_code):
    """
    Sends PDF to ocrwebservice.com (onlineocr.net backend).
    Returns an Excel file as bytes, or None on failure.
    """
    pdf_file.seek(0)
    pdf_bytes = pdf_file.read()
    
    params = {
        'language': 'english',
        'outputformat': 'xlsx',
        'pagerange': 'allpages',
    }
    
    headers = {
        'Content-Type': 'application/pdf',
    }
    
    try:
        response = requests.post(
            OCR_WEB_SERVICE_URL,
            params=params,
            headers=headers,
            data=pdf_bytes,
            auth=(username, license_code),
            timeout=60
        )
        
        if response.status_code == 401:
            st.error("OCR API Authentication failed. Check your username and license code.")
            return None
        
        result = response.json()
        
        if result.get('ErrorMessage') or result.get('OCRErrorMessage'):
            err = result.get('ErrorMessage') or result.get('OCRErrorMessage')
            st.error(f"OCR Web Service Error: {err}")
            return None
        
        # Download the output Excel file
        output_url = result.get('OutputFileUrl')
        if output_url:
            excel_response = requests.get(output_url, timeout=30)
            if excel_response.status_code == 200:
                remaining = result.get('AvailablePages', '?')
                st.info(f"OCR Web Service: {remaining} pages remaining today.")
                return excel_response.content
            else:
                st.error(f"Failed to download converted file from: {output_url}")
                return None
        else:
            st.warning("OCR Web Service returned no output file URL.")
            return None
            
    except Exception as e:
        st.error(f"OCR Web Service Request Failed: {e}")
        return None


# ============================================================
# ENGINE 2: OCR.space (Fallback)
# ============================================================
OCR_SPACE_URL = 'https://api.ocr.space/parse/image'

def ocr_via_ocrspace(pdf_file, api_key='helloworld'):
    """
    Sends PDF to OCR.space API and parses the coordinate-based result.
    Returns a DataFrame or None.
    """
    pdf_file.seek(0)
    
    payload = {
        'apikey': api_key,
        'isOverlayRequired': True,
        'detectOrientation': True,
        'scale': True,
        'OCREngine': 2,
        'isTable': True,
    }
    
    files = {'file': pdf_file}
    
    try:
        response = requests.post(OCR_SPACE_URL, files=files, data=payload, timeout=30)
        result = response.json()
        
        if result.get('IsErroredOnProcessing') or result.get('OCRExitCode') == 3:
            err = result.get('ErrorMessage')
            st.error(f"OCR.space Error: {err}")
            return None
            
        return parse_ocrspace_json(result)
        
    except Exception as e:
        st.error(f"OCR.space Request Failed: {e}")
        return None


def parse_ocrspace_json(data):
    """
    Parses OCR.space JSON using coordinate clustering.
    Rows = grouped by 'Left', Columns = identified by 'Top'.
    """
    if 'ParsedResults' not in data:
        return None
        
    rows_data = []
    
    for page in data['ParsedResults']:
        if 'TextOverlay' not in page:
            continue
            
        lines = page['TextOverlay']['Lines']
        
        def get_left(line): return line['Words'][0]['Left']
        sorted_lines = sorted(lines, key=get_left, reverse=True)
        
        # Cluster lines by Left coordinate
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
        
        # Parse each group into a row
        for group in groups:
            row_dict = {}
            desc_parts = []
            
            for line in group:
                top = line['Words'][0]['Top']
                text = line['LineText']
                
                if 100 <= top <= 400:
                    bc = text.replace(" ", "").strip()
                    if len(bc) > 3:
                        row_dict['Barcode'] = bc
                elif 500 <= top <= 600:
                    ac = text.replace(" ", "").strip()
                    if len(ac) > 3:
                        row_dict['Article'] = ac
                elif 900 <= top <= 1800:
                    desc_parts.append(text)
                elif 2000 <= top <= 2200:
                    q_text = text.replace("|", "").replace("l", "1").strip()
                    nums = re.findall(r'\d+', q_text)
                    if nums:
                        row_dict['Quantity'] = int(nums[0])

            if desc_parts:
                row_dict['Description'] = " ".join(desc_parts)
                
            if ('Description' in row_dict or 'Barcode' in row_dict or 'Article' in row_dict) and ('Quantity' in row_dict):
                rows_data.append(row_dict)
            elif 'Barcode' in row_dict and 'Description' in row_dict:
                rows_data.append(row_dict)

    if not rows_data:
        return None
        
    df = pd.DataFrame(rows_data)
    df['UOM'] = 'EA'
    return df


# ============================================================
# EXCEL PARSER — For manually converted files from onlineocr.net
# ============================================================
def parse_carrefour_excel(excel_file):
    """
    Parses an Excel file produced by onlineocr.net or ocrwebservice.com.
    Shows all columns for transparency, maps columns by header keywords.
    """
    try:
        df = pd.read_excel(excel_file, header=None)
    except Exception as e:
        st.error(f"Failed to read Excel: {e}")
        return None
    
    if df.empty:
        return None
    
    # ── DEBUG: Show raw Excel structure ──
    with st.expander("🔍 DEBUG: Raw Excel Columns (click to inspect)"):
        st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
        for col_idx in range(len(df.columns)):
            sample = df.iloc[:5, col_idx].tolist()
            st.write(f"**Col {col_idx}:** {sample}")
    
    barcode_col = None
    desc_col = None
    qty_col = None
    article_col = None
    fam_col = None
    header_row_idx = None
    
    # ── Step 1: Find header row ──
    for row_idx in range(min(15, len(df))):
        row_vals = [str(v).upper() if pd.notna(v) else '' for v in df.iloc[row_idx]]
        row_str = " ".join(row_vals)
        
        if ('BAR' in row_str) or ('QTY' in row_str) or ('ITEM' in row_str):
            header_row_idx = row_idx
            # Map each column by its header keyword
            for col_idx, val in enumerate(df.iloc[row_idx]):
                val_str = str(val).upper().strip() if pd.notna(val) else ''
                if not val_str or val_str == 'NAN':
                    continue
                
                # Exact column mapping
                if 'BAR' in val_str and 'CODE' in val_str:
                    barcode_col = col_idx
                elif val_str.startswith('BAR') and barcode_col is None:
                    barcode_col = col_idx
                elif 'QTY' in val_str and 'UC' in val_str:
                    # Specifically match "QTY UC" — this is the quantity column
                    qty_col = col_idx
                elif 'QTY' in val_str and qty_col is None:
                    qty_col = col_idx
                elif ('ITEM' in val_str and 'DESC' in val_str) or 'DESCRITION' in val_str or 'DESCRIPTION' in val_str:
                    desc_col = col_idx
                elif 'FAM' in val_str:
                    fam_col = col_idx
                elif 'SUPPLIER' in val_str and 'REF' in val_str:
                    article_col = col_idx
            break
    
    # ── Step 2: Heuristic fallback for columns not found from header ──
    
    # Find barcode column by digit count
    if barcode_col is None:
        for col_idx in range(len(df.columns)):
            col_data = df.iloc[:, col_idx].astype(str)
            barcode_count = col_data.apply(lambda x: bool(re.match(r'^\d{8,13}$', x.strip()))).sum()
            if barcode_count >= 3:
                barcode_col = col_idx
                break
    
    if barcode_col is not None:
        # Find description: column with longest average text
        if desc_col is None:
            max_text_len = 0
            for col_idx in range(len(df.columns)):
                if col_idx in [barcode_col, qty_col, article_col, fam_col]:
                    continue
                avg_len = df.iloc[:, col_idx].astype(str).apply(len).mean()
                if avg_len > max_text_len:
                    max_text_len = avg_len
                    desc_col = col_idx
        
        # Find quantity: search AFTER description, skip known columns
        if qty_col is None:
            search_start = (desc_col + 1) if desc_col is not None else 0
            for col_idx in range(search_start, len(df.columns)):
                if col_idx in [barcode_col, desc_col, article_col, fam_col]:
                    continue
                col_data = df.iloc[:, col_idx].astype(str)
                qty_count = col_data.apply(lambda x: bool(re.match(r'^\d{1,4}(\.0)?$', x.strip()))).sum()
                if qty_count >= 3:
                    qty_col = col_idx
                    break
        
        # Find article: column with leading-zero codes
        if article_col is None:
            for col_idx in range(len(df.columns)):
                if col_idx in [barcode_col, desc_col, qty_col, fam_col]:
                    continue
                col_data = df.iloc[:, col_idx].astype(str)
                art_count = col_data.apply(lambda x: bool(re.match(r'^0{2,}\d+$', x.strip()))).sum()
                if art_count >= 3:
                    article_col = col_idx
                    break
    
    if barcode_col is None and desc_col is None:
        st.error("Could not identify Barcode or Description columns in the Excel file.")
        return None
    
    # Log detected columns
    st.info(f"Columns detected → Barcode: col {barcode_col}, Desc: col {desc_col}, Qty: col {qty_col}, Article: col {article_col}, FAM: col {fam_col}")
    
    # ── Step 3: Build the result DataFrame ──
    result = pd.DataFrame()
    start_row = (header_row_idx + 1) if header_row_idx is not None else 0
    
    if barcode_col is not None:
        result['Barcode'] = df.iloc[start_row:, barcode_col].astype(str).str.strip()
    if article_col is not None:
        result['Article'] = df.iloc[start_row:, article_col].astype(str).str.strip()
    if desc_col is not None:
        result['Description'] = df.iloc[start_row:, desc_col].astype(str).str.strip()
    if qty_col is not None:
        result['Quantity'] = pd.to_numeric(df.iloc[start_row:, qty_col], errors='coerce')
    if fam_col is not None:
        result['FAM'] = df.iloc[start_row:, fam_col].astype(str).str.strip()
    
    result['UOM'] = 'EA'
    result = result.reset_index(drop=True)
    
    # ── Step 4: Clean noise rows ──
    result = clean_noise_rows(result)
    
    return result if not result.empty else None


# ============================================================
# NOISE FILTER — Remove header/footer rows from any source
# ============================================================
NOISE_KEYWORDS = [
    'MUSCAT', 'OMAN', 'SEEB', 'TERMS', 'SPECIAL CONDITIONS',
    'P.O. BOX', 'PURCHASE ORDER', 'SUPPLIER', 'BAR CODE',
    'CURRENCY', 'DEPARTMENT', 'SECTION', 'DELIVERED',
    'DELIVERY DATE', 'DEADLINE', 'CONTACT', 'PAGE :',
    'MAJID AL FUTTAIM', 'HYPERMARKETS', 'TEL :', 'FAX :',
    'Retail.Any', 'assignment', 'null and void', 'force and effect',
    '-----', '=====', 'CHEQUE', 'PAYMENT',
    'INVOICE', 'DELIVERIES', 'PARTIAL DELIVER', 'SIGNATURE',
    'COMPUTER GENERATED', 'ORDER DATE', 'VAT REGISTERED',
    'OMANI RIALS', 'SUP TRN', 'KHIMJI RAMDAS',
    'TRN #', 'DAYS END OF MONTH',
]

def clean_noise_rows(df):
    """Remove rows that are clearly header/footer noise, not product data."""
    if df.empty:
        return df
    
    def is_noise(row):
        # Check Description for noise keywords
        desc = str(row.get('Description', '')).upper().strip()
        barcode = str(row.get('Barcode', '')).strip()
        
        # Normalize NaN values
        if barcode.lower() == 'nan':
            barcode = ''
        if desc == 'NAN' or desc == 'NONE':
            desc = ''
        
        # If description matches noise keywords
        for keyword in NOISE_KEYWORDS:
            if keyword.upper() in desc:
                return True
        
        # Also check barcode field for noise keywords (sometimes entire header ends up there)
        if barcode:
            barcode_upper = barcode.upper()
            for keyword in NOISE_KEYWORDS:
                if keyword.upper() in barcode_upper:
                    return True
        
        # If barcode is present and clearly not a barcode (all letters, no digits)
        if barcode and not any(c.isdigit() for c in barcode):
            return True
        
        # If barcode looks like location/text (e.g., "MUSCAT")
        if barcode and barcode.isalpha() and len(barcode) > 2:
            return True
        
        # If barcode contains a decimal point → probably a total/summary, not a real barcode
        if barcode and '.' in barcode and not desc:
            return True
            
        # If BOTH description and barcode are empty/missing → noise
        if not desc and not barcode:
            return True
        
        return False
    
    mask = df.apply(is_noise, axis=1)
    cleaned = df[~mask].reset_index(drop=True)
    
    dropped = mask.sum()
    if dropped > 0:
        st.info(f"Filtered out {dropped} header/footer noise rows.")
    
    return cleaned


# ============================================================
# MAIN ENTRY POINT
# ============================================================
def get_carrefour_data(pdf_file, ocr_engine='ocrspace', api_key='helloworld', 
                       ws_username=None, ws_license=None):
    """
    Main function to extract data from a Carrefour fax PDF.
    
    Args:
        pdf_file: The uploaded PDF file object.
        ocr_engine: 'webservice' for ocrwebservice.com, 'ocrspace' for OCR.space.
        api_key: API key for OCR.space (only used if engine='ocrspace').
        ws_username: Username for ocrwebservice.com.
        ws_license: License code for ocrwebservice.com.
    
    Returns:
        pandas DataFrame or None.
    """
    
    if ocr_engine == 'webservice' and ws_username and ws_license:
        # Primary: ocrwebservice.com → Excel output
        st.info("Using ocrwebservice.com (onlineocr.net engine)...")
        excel_bytes = ocr_via_webservice(pdf_file, ws_username, ws_license)
        if excel_bytes:
            df = parse_carrefour_excel(io.BytesIO(excel_bytes))
            if df is not None:
                return df
            else:
                st.warning("Failed to parse Excel from web service. Falling back to OCR.space...")
    
    # Fallback: OCR.space
    st.info("Using OCR.space engine...")
    df = ocr_via_ocrspace(pdf_file, api_key=api_key)
    if df is not None:
        df = clean_noise_rows(df)
    return df
