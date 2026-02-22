import streamlit as st
import pdfplumber
import pandas as pd
import numpy as np
import re
import io
import requests
from difflib import SequenceMatcher
from ocr_lib import get_carrefour_data, parse_carrefour_excel
import os
import streamlit.components.v1 as components

# Set up the web page
st.set_page_config(page_title="Universal PO & Multi-Layer Mapper", layout="wide")
st.title("📄 Universal PO Converter & Multi-Layer Mapper")
st.markdown("Works for **LuLu, Nesto, Talabat, Carrefour & WHSmith**! Upload the **PDF PO** or paste WHSmith email text. Master File & Order Forms are built-in.")

# --- HARDCODED REFERENCE FILES ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MASTER_FILE_PATH = os.path.join(SCRIPT_DIR, "Bel Article Master 12.25.xlsx")
LULU_ORDER_PATH = os.path.join(SCRIPT_DIR, "Lulu Order Form 8.24 (1).xlsx")
NESTO_ORDER_PATH = os.path.join(SCRIPT_DIR, "Nesto Order Form 8.24.xlsx")

# --- HARDCODED OCR CREDENTIALS (ocrwebservice.com) ---
# Uses webservice engine for best quality; OCR.space as automatic fallback.
ocr_engine = 'webservice'
ws_username = 'krtrial'
ws_license = '96C8A75B-0EDC-42E2-8594-3B95D6DBCB27'
ocr_api_key = 'helloworld'  # Fallback OCR.space key

with st.sidebar:
    st.header("OCR Settings (Carrefour Fax)")
    st.success("✅ OCR is pre-configured and ready to use.")
    st.markdown("Scanned PDFs (e.g. Carrefour fax) will be auto-processed using the built-in OCR engine.")
    st.divider()
    st.markdown("**Tip:** For best results, convert the fax to Excel on [onlineocr.net](https://www.onlineocr.net) and upload the Excel directly as your PO file.")

# --- FUNCTION 1: CLEAN KEYS FOR BARCODE MAPPING ---
def clean_key(val):
    if pd.isna(val):
        return ""
    try:
        return str(int(float(val)))
    except ValueError:
        return str(val).strip().upper()

# --- FUNCTION 2: FOOLPROOF TEXT CLEANER FOR DESCRIPTION FALLBACK ---
def clean_desc(val):
    if pd.isna(val):
        return ""
    cleaned_text = str(val).strip().upper()
    return re.sub(r'\s+', ' ', cleaned_text)

# --- FUNCTION 2b: FUZZY-FRIENDLY TEXT CLEANER ---
def clean_desc_fuzzy(val):
    """Clean description for fuzzy matching.
    Removes pack info like (12), (24), punctuation, extra spaces.
    Keeps only letters, digits, and spaces for comparison.
    """
    if pd.isna(val):
        return ""
    s = str(val).strip().upper()
    s = re.sub(r'\([^)]*\)', '', s)       # Remove (12), (24x500ML), etc.
    s = re.sub(r'[^A-Z0-9\s]', ' ', s)    # Replace punctuation with space
    s = re.sub(r'\s+', ' ', s).strip()     # Collapse whitespace
    return s

# --- FUNCTION 2c: FUZZY DESCRIPTION MATCHING ---
def fuzzy_match_desc(target, candidates, threshold=0.70):
    """Find best matching description from candidates list.
    
    Args:
        target: Clean description string to match
        candidates: List of dicts with keys 'desc', 'article', 'row', 'price', 'raw_desc'
        threshold: Minimum similarity ratio (0.0 - 1.0)
    
    Returns:
        (article, row, raw_desc, score) tuple or None
    """
    if not target or not candidates:
        return None
    
    best_score = 0
    best_match = None
    
    for candidate in candidates:
        try:
            cand_desc = candidate['desc']
            if not cand_desc:
                continue
            score = SequenceMatcher(None, target, cand_desc).ratio()
            if score > best_score:
                best_score = score
                best_match = candidate
        except Exception:
            continue
    
    if best_score >= threshold and best_match:
        return (best_match['article'], best_match['row'], best_match['raw_desc'], best_score)
    return None

# --- FUNCTION 2d: PRICE + FUZZY DESCRIPTION MATCHING ---
def fuzzy_match_with_price(target_desc, target_price, candidates, 
                            price_tolerance=0.05, desc_threshold=0.60):
    """Find best match by first filtering on price, then fuzzy matching desc.
    
    Args:
        target_desc: Clean description string to match
        target_price: Numeric price from the PO
        candidates: List of dicts with keys 'desc', 'article', 'row', 'price', 'raw_desc'
        price_tolerance: Price tolerance ratio (0.05 = ±5%)
        desc_threshold: Minimum description similarity
    
    Returns:
        (article, row, raw_desc, score) tuple or None
    """
    if not target_desc or not target_price or not candidates:
        return None
    
    try:
        target_price = float(target_price)
    except (ValueError, TypeError):
        return None
    
    if target_price <= 0:
        return None
    
    # Filter candidates by price (within tolerance)
    price_matches = []
    for candidate in candidates:
        try:
            cand_price = candidate.get('price')
            if cand_price is None or cand_price <= 0:
                continue
            # Check if prices are within tolerance
            price_diff = abs(cand_price - target_price) / target_price
            if price_diff <= price_tolerance:
                price_matches.append(candidate)
        except (ValueError, TypeError):
            continue
    
    if not price_matches:
        return None
    
    # Among price-matched candidates, find best description match
    return fuzzy_match_desc(target_desc, price_matches, threshold=desc_threshold)
# --- FUNCTION 2e: PARSE WHSMITH EMAIL TEXT ---
def parse_whsmith_text(text):
    """Parse WHSmith order data pasted from email.
    
    The data has a repeating pattern per product:
        Item_No (header line)
        Barcode (header line)
        Item_Description (header line)
        Order WK## (header line)
        
        item_no_value
        barcode_value
        description_value
        qty_value (e.g. '4 OTR', '24POUCH', '36 BAGS')
    """
    lines = [l.strip() for l in text.strip().split('\n') if l.strip()]
    
    # Remove header lines (Item_No, Barcode, Item_Description, Order WK##)
    # Find where data starts (first line that looks like a number/item code)
    data_start = 0
    for i, line in enumerate(lines):
        # Skip header labels
        if line.lower() in ('item_no', 'barcode', 'item_description') or line.lower().startswith('order '):
            data_start = i + 1
            continue
        # First non-header line
        if i >= data_start:
            break
    
    data_lines = lines[data_start:]
    
    # Parse groups of 4 lines: item_no, barcode, description, qty
    rows = []
    i = 0
    while i < len(data_lines):
        # Need at least 1 more line (some may be on same line)
        item_no = data_lines[i].strip() if i < len(data_lines) else ''
        barcode = data_lines[i+1].strip() if i+1 < len(data_lines) else ''
        desc = data_lines[i+2].strip() if i+2 < len(data_lines) else ''
        qty_raw = data_lines[i+3].strip() if i+3 < len(data_lines) else ''
        
        # Parse qty+UOM from strings like '4 OTR', '24POUCH', '36 BAGS'
        qty_match = re.match(r'(\d+)\s*(OTR|POUCH|BAGS|EA|PCS|CTN|CS|CV)?', qty_raw, re.IGNORECASE)
        if qty_match:
            qty = int(qty_match.group(1))
            uom = qty_match.group(2).upper() if qty_match.group(2) else 'EA'
        else:
            qty = 0
            uom = ''
        
        if item_no and barcode:
            rows.append({
                'Article': item_no,
                'Barcode': barcode,
                'Description': desc,
                'Quantity': qty,
                'UOM': uom,
                'Raw Qty': qty_raw
            })
        
        i += 4
    
    if not rows:
        return None
    
    df = pd.DataFrame(rows)
    df['True Quantity'] = df['Quantity']  # Will be overridden after OTR conversion
    return df

# --- FUNCTION 3: PROCESS PDF AND CALCULATE TRUE QUANTITY ---
def process_pdf(pdf_file, api_key=None, ocr_engine='ocrspace', ws_username=None, ws_license=None):
    all_rows = []
    
    try:
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                table = page.extract_table()
                if table:
                    all_rows.extend(table)
                    
        if not all_rows:
            # Fallback to OCR if no text found (Scanned PDF)
            st.warning("No text found in PDF. Attempting OCR (Optical Character Recognition)...")
            df_ocr = get_carrefour_data(
                pdf_file,
                ocr_engine=ocr_engine,
                api_key=api_key if api_key else 'helloworld',
                ws_username=ws_username,
                ws_license=ws_license
            )
            if df_ocr is not None and not df_ocr.empty:
                st.success(f"OCR Successful! Extracted {len(df_ocr)} rows.")
                return df_ocr
            else:
                st.error("OCR failed to extract data. Check your API credentials.")
                return None

        # --- SMART COLUMN ALIGNMENT ---
        # pdfplumber sometimes drops unnamed/empty columns on later pages.
        # E.g. Nesto Page 1 has 18 cols (with an unnamed col 6), but Page 2 has 17.
        # Simple end-padding would shift Barcode, Qty, UOM etc. left by 1.
        # FIX: Find the header row FIRST, then for shorter rows, insert None
        # at the positions of unnamed header columns to re-align everything.
        
        # Step 1: Find the header row in the raw list
        header_idx = None
        for i, row in enumerate(all_rows):
            row_str = " ".join([str(val).upper().replace('\n', ' ') for val in row if val is not None])
            # Match LuLu (QUANTITY + UOM), Nesto (ORD.QTY + UNIT), or Talabat (QTY + UNIT COST / BARCODE + PRODUCT)
            has_qty = "QUANTITY" in row_str or "ORD.QTY" in row_str or "QTY" in row_str
            has_unit = "UOM" in row_str or "UNIT" in row_str
            has_product = "BARCODE" in row_str and "PRODUCT" in row_str
            if (has_qty and has_unit) or has_product:
                header_idx = i
                break
        
        if header_idx is None:
            # DEBUG: Show what we searched so we can diagnose
            st.warning("Could not automatically find the Quantity/UOM headers. Returning raw data.")
            if all_rows:
                for di in range(min(3, len(all_rows))):
                    dbg_str = " ".join([str(val).upper().replace('\n', ' ') for val in all_rows[di] if val is not None])
                    st.text(f"  Debug Row {di}: {dbg_str[:120]}")
            # Fallback: pad at end and return raw
            max_cols = max(len(r) for r in all_rows)
            all_rows = [r + [None] * (max_cols - len(r)) for r in all_rows]
            df = pd.DataFrame(all_rows)
            df = df.map(lambda x: str(x).replace('\n', ' ').strip() if pd.notnull(x) else x)
            return df
        
        header_row = all_rows[header_idx]
        header_len = len(header_row)
        
        # Step 2: Find positions of unnamed/empty columns in the header
        empty_col_indices = [i for i, h in enumerate(header_row) if h is None or str(h).strip() == '']
        
        # Step 3: For each row shorter than header, insert None at empty-column positions
        aligned_rows = []
        for row in all_rows:
            if len(row) < header_len:
                diff = header_len - len(row)
                row = list(row)
                # Insert None at the first 'diff' empty header positions (right to left)
                for insert_pos in sorted(empty_col_indices[:diff], reverse=True):
                    row.insert(insert_pos, None)
                # If still short for any reason, pad at end
                row = row + [None] * (header_len - len(row))
            elif len(row) > header_len:
                row = row[:header_len]
            aligned_rows.append(row)

        df = pd.DataFrame(aligned_rows)
        df = df.map(lambda x: str(x).replace('\n', ' ').strip() if pd.notnull(x) else x)
        
        # Use the header row we already found
        df.columns = df.iloc[header_idx]
        # Clean newlines from column names (Talabat has 'Unit\nCost', 'Amt.\nExcl.\nVAT' etc.)
        df.columns = [str(c).replace('\n', ' ').strip() if pd.notna(c) else c for c in df.columns]
        df = df.iloc[header_idx + 1:].reset_index(drop=True)
        
        first_col_name = df.columns[0]
        if pd.notnull(first_col_name):
            df = df[df[first_col_name] != first_col_name]
            
        df = df.dropna(how='all')
        df = df.loc[:, df.columns.notna()]

        # DYNAMIC TRUE QUANTITY CALCULATION (Adapts to LuLu & Nesto columns)
        def calculate_true_quantity(row):
            try:
                # Dynamically fetch Quantity (LuLu = 'Quantity', Nesto = 'Ord.Qty', Talabat = 'Qty')
                if 'Quantity' in row.index:
                    qty_val = row.get('Quantity')
                elif 'Ord.Qty' in row.index:
                    qty_val = row.get('Ord.Qty', 0)
                elif 'Qty' in row.index:
                    qty_val = row.get('Qty', 0)
                else:
                    qty_val = 0
                qty = pd.to_numeric(qty_val, errors='coerce')
                if pd.isna(qty):
                    qty = 0
                    
                # Dynamically fetch UOM (LuLu = 'UOM', Nesto = 'UnitConv.' or 'Unit')
                if 'UOM' in row.index:
                    uom_val = row.get('UOM', '')
                elif 'UnitConv.' in row.index:
                    uom_val = row.get('UnitConv.', '')
                else:
                    uom_val = row.get('Unit', '')
                uom = str(uom_val).upper()
                
                match = re.search(r'(\d+)\s*EA', uom)
                multiplier = float(match.group(1)) if match else 1.0
                    
                total_qty = qty * multiplier
                
                if total_qty.is_integer():
                    return int(total_qty)
                return float(total_qty)
            except Exception:
                return None 

        # Apply True Quantity for any valid PO format
        df['True Quantity'] = df.apply(calculate_true_quantity, axis=1)
        
        # --- NET TOTAL calculation (LuLu = Total Value - Tax; Talabat = Amt Excl VAT directly) ---
        # Check if Talabat's Amt. Excl. VAT column exists (already net total)
        amt_excl_col = None
        for c in df.columns:
            cs = str(c).upper().replace('\n', ' ') if pd.notna(c) else ''
            if 'AMT' in cs and 'EXCL' in cs and 'VAT' in cs:
                amt_excl_col = c
                break
        
        if amt_excl_col:
            # Talabat: net total is already given as Amt. Excl. VAT
            df['Net Total'] = pd.to_numeric(df[amt_excl_col], errors='coerce').round(3)
            st.info(f"💰 Net Total from **{amt_excl_col.replace(chr(10), ' ')}**")
        else:
            # LuLu: calculate Total Value - Tax Val
            total_col = None
            tax_col = None
            for c in df.columns:
                cs = str(c).upper() if pd.notna(c) else ''
                if 'TOTAL' in cs and ('VALUE' in cs or 'WT' in cs or 'VAL' in cs):
                    total_col = c
                if 'TAX' in cs and ('VAL' in cs or 'AMT' in cs or 'AMOUNT' in cs):
                    tax_col = c
        
        if not amt_excl_col and total_col and tax_col:
            try:
                total_vals = pd.to_numeric(df[total_col], errors='coerce').fillna(0)
                tax_vals = pd.to_numeric(df[tax_col], errors='coerce').fillna(0)
                df['Net Total'] = (total_vals - tax_vals).round(3)
                st.info(f"💰 Net Total calculated: **{total_col}** − **{tax_col}**")
            except Exception as e:
                st.warning(f"Could not calculate Net Total: {e}")
        
        # --- LULU-SPECIFIC: Per-piece Unit Price for price matching ---
        # Gross/PU is price per packaging unit (e.g., per carton of 6)
        # For accurate matching vs Master File (which has per-piece prices),
        # we need:  Unit Price = Gross/PU ÷ pieces_per_carton
        gross_col = None
        for c in df.columns:
            cs = str(c).upper() if pd.notna(c) else ''
            if 'GROSS' in cs and ('PU' in cs or 'UNIT' in cs):
                gross_col = c
                break
        if gross_col is None:
            for c in df.columns:
                cs = str(c).upper() if pd.notna(c) else ''
                if 'UNIT' in cs and ('COST' in cs or 'PRICE' in cs):
                    gross_col = c
                    break
        
        if gross_col:
            def calc_unit_price(row):
                try:
                    gross = pd.to_numeric(row.get(gross_col, 0), errors='coerce')
                    if pd.isna(gross) or gross == 0:
                        return None
                    
                    # Get UOM multiplier (same logic as calculate_true_quantity)
                    if 'UOM' in row.index:
                        uom_val = row.get('UOM', '')
                    elif 'UnitConv.' in row.index:
                        uom_val = row.get('UnitConv.', '')
                    else:
                        uom_val = row.get('Unit', '')
                    uom = str(uom_val).upper()
                    
                    match = re.search(r'(\d+)\s*EA', uom)
                    multiplier = float(match.group(1)) if match else 1.0
                    
                    unit_price = gross / multiplier
                    return round(unit_price, 3)
                except Exception:
                    return None
            
            df['Unit Price'] = df.apply(calc_unit_price, axis=1)
            st.info(f"💰 Unit Price calculated: **{gross_col}** ÷ UOM multiplier")
        
        return df

    except Exception as e:
        st.error(f"An unexpected error occurred while processing the PDF: {e}")
        return None

# --- FUNCTION 4: APPLY MULTI-LAYER MAPPING ---
def apply_mappings(df_po, master_override=None, lulu_override=None, nesto_override=None):
    try:
        # ==========================================
        # 1. LOAD MASTER FILE (ZDET-PRICE)
        # ==========================================
        if master_override is not None:
            # User uploaded a newer Master File
            st.info("📤 Using **uploaded** Master File (override)")
            try:
                if master_override.name.endswith('.csv'):
                    df_master_raw = pd.read_csv(master_override, header=None)
                else:
                    df_master_raw = pd.read_excel(master_override, sheet_name='ZDET-PRICE', header=None)
            except ValueError:
                st.error("🚨 Could not find 'ZDET-PRICE' sheet in uploaded Master File.")
                return None
        else:
            # Use hardcoded default
            if not os.path.exists(MASTER_FILE_PATH):
                st.error(f"🚨 Master File not found: {MASTER_FILE_PATH}")
                return None
            try:
                df_master_raw = pd.read_excel(MASTER_FILE_PATH, sheet_name='ZDET-PRICE', header=None)
            except ValueError:
                st.error("🚨 Could not find 'ZDET-PRICE' sheet in the Master File.")
                return None

        df_master = None
        for idx, row in df_master_raw.iterrows():
            if idx > 50: break
            row_str = " ".join([str(val).upper() for val in row.values if pd.notnull(val)])
            if "ARTICLE" in row_str and "GTIN" in row_str:
                df_master_raw.columns = df_master_raw.iloc[idx]
                df_master = df_master_raw.iloc[idx + 1:].reset_index(drop=True)
                break
                
        if df_master is not None:
            df_master = df_master.loc[:, ~df_master.columns.duplicated()]
            df_master.columns = [str(c).strip() for c in df_master.columns]
            df_master['GTIN'] = df_master['GTIN'].apply(clean_key)
            
            desc_col_name = "Material Description" if "Material Description" in df_master.columns else "Description"
            if desc_col_name in df_master.columns:
                df_master[desc_col_name] = df_master[desc_col_name].apply(clean_desc)
            
            # Find Master File price column
            master_price_col = None
            for c in df_master.columns:
                cu = c.upper()
                if 'SELLING' in cu and 'PRICE' in cu:
                    master_price_col = c
                    break
            if master_price_col is None:
                for c in df_master.columns:
                    cu = c.upper()
                    if 'NET' in cu and 'PRICE' in cu:
                        master_price_col = c
                        break
            if master_price_col is None:
                for c in df_master.columns:
                    cu = c.upper()
                    if 'PRICE' in cu or 'COST' in cu:
                        master_price_col = c
                        break
                
            # Store (kr_code, source_row) tuples for provenance tracking
            master_gtin_dict = {row['GTIN']: (row['Article'], idx + 2)
                                for idx, row in df_master.iterrows()
                                if row['GTIN'] != ""}
            master_desc_dict = {row[desc_col_name]: (row['Article'], idx + 2)
                                for idx, row in df_master.iterrows()
                                if desc_col_name in df_master.columns and row.get(desc_col_name, "") != ""}
            
            # Build master list for fuzzy matching: [(clean_desc, article, row, price)]
            master_fuzzy_list = []
            for idx, row in df_master.iterrows():
                desc_val = str(row.get(desc_col_name, '')).strip()
                if desc_val and desc_val.upper() != 'NAN':
                    price_val = None
                    if master_price_col:
                        try:
                            price_val = float(row.get(master_price_col, 0))
                        except (ValueError, TypeError):
                            price_val = None
                    master_fuzzy_list.append({
                        'desc': clean_desc_fuzzy(desc_val),
                        'article': row['Article'],
                        'row': idx + 2,
                        'price': price_val,
                        'raw_desc': desc_val
                    })
            
            if master_price_col:
                st.success(f"✅ Master Mapping loaded! (Price col: {master_price_col[:30]})")
            else:
                st.success("✅ Master Mapping (ZDET-PRICE) loaded!")
        else:
            st.error("🚨 Missing GTIN/Article headers in Master File.")
            return None

        # ==========================================
        # 2. LOAD ORDER FORMS (LULU + NESTO DUAL FALLBACK) — HARDCODED
        # ==========================================
        order_barcode_dict, order_retailer_dict, order_desc_dict = {}, {}, {}
        order_fuzzy_list = []
        
        # Helper: load one order form file and extract lookup dicts
        def load_order_form(filepath, label):
            if not os.path.exists(filepath):
                st.warning(f"⚠️ {label} not found: {os.path.basename(filepath)}")
                return None, {}, {}, {}, []
            
            df_order_dict = pd.read_excel(filepath, sheet_name=None, header=None)
            
            df_order = None
            for sheet_name, df_sheet in df_order_dict.items():
                for idx, row in df_sheet.iterrows():
                    if idx > 100: break
                    row_str = " ".join([str(val).upper() for val in row.values if pd.notnull(val)])
                    if ("BARCODES" in row_str or "BARCODE" in row_str) and ("KR CODE" in row_str or "SAP CODE" in row_str):
                        df_sheet.columns = df_sheet.iloc[idx]
                        df_order = df_sheet.iloc[idx + 1:].reset_index(drop=True)
                        break
                if df_order is not None: break
            
            if df_order is None:
                st.warning(f"⚠️ Could not find headers in {label}")
                return None, {}, {}, {}, []
            
            df_order = df_order.loc[:, ~df_order.columns.duplicated()]
            df_order.columns = [str(c).strip().upper() for c in df_order.columns]
            
            of_barcode = 'BARCODES' if 'BARCODES' in df_order.columns else ('BARCODE (PER UNIT)' if 'BARCODE (PER UNIT)' in df_order.columns else None)
            of_kr = 'KR CODE' if 'KR CODE' in df_order.columns else ('SAP CODE' if 'SAP CODE' in df_order.columns else None)
            of_retailer = 'LULU CODE' if 'LULU CODE' in df_order.columns else ('NESTO CODE' if 'NESTO CODE' in df_order.columns else None)
            of_desc = 'PRODUCT' if 'PRODUCT' in df_order.columns else ('PRODUCT DESCRIPTION' if 'PRODUCT DESCRIPTION' in df_order.columns else None)
            
            bc_dict, ret_dict, desc_dict, fuzzy_list = {}, {}, {}, []
            
            if of_barcode and of_kr:
                df_order[of_barcode] = df_order[of_barcode].apply(clean_key)
                bc_dict = {row[of_barcode]: (row[of_kr], idx + 2)
                           for idx, row in df_order.iterrows()
                           if row[of_barcode] != ""}
            
            if of_retailer and of_kr:
                df_order[of_retailer] = df_order[of_retailer].apply(clean_key)
                ret_dict = {row[of_retailer]: (row[of_kr], idx + 2)
                            for idx, row in df_order.iterrows()
                            if row[of_retailer] != ""}
            
            if of_desc and of_kr:
                df_order[of_desc] = df_order[of_desc].apply(clean_desc)
                desc_dict = {row[of_desc]: (row[of_kr], idx + 2)
                             for idx, row in df_order.iterrows()
                             if row[of_desc] != ""}
                for idx, row in df_order.iterrows():
                    desc_val = str(row.get(of_desc, '')).strip()
                    if desc_val and desc_val.upper() != 'NAN':
                        fuzzy_list.append({
                            'desc': clean_desc_fuzzy(desc_val),
                            'article': row.get(of_kr, ''),
                            'row': idx + 2,
                            'price': None,
                            'raw_desc': desc_val
                        })
            
            return df_order, bc_dict, ret_dict, desc_dict, fuzzy_list
        
        # Load Lulu Order Form (use override if uploaded, else hardcoded)
        if lulu_override is not None:
            st.info("📤 Using **uploaded** Lulu Order Form (override)")
            # Save to temp file for load_order_form
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp:
                tmp.write(lulu_override.read())
                lulu_tmp_path = tmp.name
            _, lulu_bc, lulu_ret, lulu_desc, lulu_fuzzy = load_order_form(lulu_tmp_path, "Lulu Order Form (uploaded)")
        else:
            _, lulu_bc, lulu_ret, lulu_desc, lulu_fuzzy = load_order_form(LULU_ORDER_PATH, "Lulu Order Form")
        
        # Load Nesto Order Form (use override if uploaded, else hardcoded)
        if nesto_override is not None:
            st.info("📤 Using **uploaded** Nesto Order Form (override)")
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp:
                tmp.write(nesto_override.read())
                nesto_tmp_path = tmp.name
            _, nesto_bc, nesto_ret, nesto_desc, nesto_fuzzy = load_order_form(nesto_tmp_path, "Nesto Order Form (uploaded)")
        else:
            _, nesto_bc, nesto_ret, nesto_desc, nesto_fuzzy = load_order_form(NESTO_ORDER_PATH, "Nesto Order Form")
        
        # Keep separate for provenance tracking (L2-1 Lulu vs L2-2 Nesto)
        # Combined dicts for L5 (exact desc) and L7 (fuzzy) where source doesn't matter as much
        order_desc_dict = {**nesto_desc, **lulu_desc}
        order_fuzzy_list = lulu_fuzzy + nesto_fuzzy
        
        lulu_count = len(lulu_bc) + len(lulu_ret) + len(lulu_desc)
        nesto_count = len(nesto_bc) + len(nesto_ret) + len(nesto_desc)
        st.success(f"✅ Dual Order Form loaded! Lulu: {lulu_count} entries, Nesto: {nesto_count} entries, {len(order_fuzzy_list)} fuzzy items")

        # ==========================================
        # 3. APPLY MULTI-LAYER MAPPING TO PO
        # ==========================================
        # Dynamically identify PO description column (LuLu/Nesto/Talabat)
        po_desc_col = None
        for candidate in ['Article Description + Add.Info', 'Description', 'Product', 'ITEM DESCRITION']:
            if candidate in df_po.columns:
                po_desc_col = candidate
                break
        
        # Find PO price column (LuLu=Gross/PU, Nesto=Unit.Cost, Carrefour=P.PRI B.TAX)
        po_price_col = None
        for c in df_po.columns:
            cu = str(c).upper()
            if 'GROSS/PU' in cu or 'UNIT.COST' in cu or 'UNIT COST' in cu or 'P.PRI' in cu or 'PPRI' in cu:
                po_price_col = c
                break
        if po_price_col is None:
            for c in df_po.columns:
                cu = str(c).upper()
                if 'PRICE' in cu or 'COST' in cu:
                    po_price_col = c
                    break
        
        # Clean PO matching keys safely
        # Handle different barcode column names
        barcode_col = None
        for bcol in ['Barcode', 'BARCODE', 'BAR CODE']:
            if bcol in df_po.columns:
                barcode_col = bcol
                break
        if barcode_col:
            df_po['Barcode'] = df_po[barcode_col]  # Normalize to 'Barcode'
            df_po['Barcode_Clean'] = df_po[barcode_col].apply(clean_key)
        else:
            df_po['Barcode_Clean'] = ""
        
        # Handle different article/supplier ref column names
        article_col = None
        for acol in ['Article', 'Supplier SKU', 'SUPPLIER REF', 'SKU']:
            if acol in df_po.columns:
                article_col = acol
                break
        if article_col:
            df_po['Article'] = df_po[article_col]  # Normalize to 'Article'
            df_po['Article Code'] = df_po[article_col].apply(clean_key)
        else:
            df_po['Article Code'] = ""
        
        # Clean descriptions for matching
        if po_desc_col: df_po['Clean PO Desc'] = df_po[po_desc_col].apply(clean_desc)
        else: df_po['Clean PO Desc'] = ""
        
        # Also build fuzzy-friendly PO description
        if po_desc_col: df_po['Fuzzy PO Desc'] = df_po[po_desc_col].apply(clean_desc_fuzzy)
        else: df_po['Fuzzy PO Desc'] = ""
        
        # Extract PO price for each row
        # Prefer 'Unit Price' (per-piece, calculated in process_pdf) over raw Gross/PU
        if 'Unit Price' in df_po.columns:
            df_po['PO_Price'] = pd.to_numeric(df_po['Unit Price'], errors='coerce')
            st.info("📊 Using **per-piece Unit Price** for price matching (Gross/PU ÷ UOM)")
        elif po_price_col:
            df_po['PO_Price'] = pd.to_numeric(df_po[po_price_col], errors='coerce')
        else:
            df_po['PO_Price'] = np.nan


        # The Waterfall Mapping Function (returns tuple for provenance)
        def find_kr_code(row):
            try:
                barcode = row.get('Barcode_Clean', "")
                po_article = row.get('Article Code', "")
                desc = row.get('Clean PO Desc', "")
                fuzzy_desc = row.get('Fuzzy PO Desc', "")
                po_price = row.get('PO_Price', None)
                if pd.isna(po_price):
                    po_price = None
                
                # Layer 1: Master File Barcode (GTIN)
                if barcode != "" and barcode in master_gtin_dict:
                    kr, src_row = master_gtin_dict[barcode]
                    return pd.Series([kr, "L1: Master (GTIN)", barcode, src_row])
                    
                # Layer 2-1: Lulu Order Form Barcode
                if barcode != "" and barcode in lulu_bc:
                    kr, src_row = lulu_bc[barcode]
                    return pd.Series([kr, "L2-1: Lulu (Barcode)", barcode, src_row])
                
                # Layer 2-2: Nesto Order Form Barcode
                if barcode != "" and barcode in nesto_bc:
                    kr, src_row = nesto_bc[barcode]
                    return pd.Series([kr, "L2-2: Nesto (Barcode)", barcode, src_row])
                    
                # Layer 3-1: Lulu Retailer Code
                if po_article != "" and po_article in lulu_ret:
                    kr, src_row = lulu_ret[po_article]
                    return pd.Series([kr, "L3-1: Lulu (Retailer)", po_article, src_row])
                
                # Layer 3-2: Nesto Retailer Code
                if po_article != "" and po_article in nesto_ret:
                    kr, src_row = nesto_ret[po_article]
                    return pd.Series([kr, "L3-2: Nesto (Retailer)", po_article, src_row])
                    
                # Layer 4: Master File Exact Description
                if desc != "" and desc in master_desc_dict:
                    kr, src_row = master_desc_dict[desc]
                    return pd.Series([kr, "L4: Master (Desc)", desc[:40], src_row])
                    
                # Layer 5: Order Form Exact Description
                if desc != "" and desc in order_desc_dict:
                    kr, src_row = order_desc_dict[desc]
                    return pd.Series([kr, "L5: Order (Desc)", desc[:40], src_row])
                
                # Layer 6: Master File Fuzzy Description (70%+ similarity)
                if fuzzy_desc != "":
                    best = fuzzy_match_desc(fuzzy_desc, master_fuzzy_list, threshold=0.70)
                    if best:
                        kr, src_row, matched, score = best
                        return pd.Series([kr, f"L6: Master (Fuzzy {score:.0%})", matched[:35], src_row])
                
                # Layer 7: Order Form Fuzzy Description (70%+ similarity)
                if fuzzy_desc != "" and order_fuzzy_list:
                    best = fuzzy_match_desc(fuzzy_desc, order_fuzzy_list, threshold=0.70)
                    if best:
                        kr, src_row, matched, score = best
                        return pd.Series([kr, f"L7: Order (Fuzzy {score:.0%})", matched[:35], src_row])
                
                # Layer 8: Master File Price + Fuzzy Desc (price match ±5%, desc 60%+)
                if po_price is not None and po_price > 0 and fuzzy_desc != "":
                    best = fuzzy_match_with_price(fuzzy_desc, po_price, master_fuzzy_list, 
                                                  price_tolerance=0.05, desc_threshold=0.60)
                    if best:
                        kr, src_row, matched, score = best
                        return pd.Series([kr, f"L8: Master (Price+Fuzzy {score:.0%})", matched[:35], src_row])
                    
                return pd.Series(["Not Found", "-", "-", "-"])
            except Exception as e:
                return pd.Series(["Error", str(e)[:30], "-", "-"])

        df_po[['KR CODE', 'Match Source', 'Match Key', 'Source Row']] = df_po.apply(find_kr_code, axis=1)

        # Output correct columns based on file type
        desired_columns = ['KR CODE', 'True Quantity', 'Net Total', 'Unit Price', 'Match Source', 'Match Key', 'Source Row', 'FAM', 'Barcode', 'Article']
        if po_desc_col: desired_columns.append(po_desc_col)
        
        final_cols = [col for col in desired_columns if col in df_po.columns]
        result = df_po[final_cols].reset_index(drop=True)
        result.index = result.index + 1  # Start row numbers from 1
        return result
        
    except Exception as e:
        st.error(f"An error occurred while applying mappings: {e}")
        return None

# --- UI LAYOUT ---
# Anchor: top of page (for "Back to Top" button)
st.markdown('<div id="top-anchor"></div>', unsafe_allow_html=True)
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📂 Upload PO File")
    po_file = st.file_uploader("Upload PO (PDF or Excel from OCR)", type=["pdf", "xlsx", "xls"])

with col2:
    st.subheader("📧 Or Paste WHSmith Email")
    whsmith_text = st.text_area(
        "Paste WHSmith order here:",
        height=200,
        placeholder="Item_No\nBarcode\nItem_Description\nOrder WK23\n\n1001010023\n87108408\nFruittella Strawberry\n4 OTR\n..."
    )

st.divider()

# Sidebar: Built-in files status + optional override uploaders
with st.sidebar:
    st.divider()
    st.subheader("📁 Reference Files")
    st.caption("Built-in files are used by default. Upload newer versions to override.")
    
    # Master File
    if os.path.exists(MASTER_FILE_PATH):
        st.success(f"✅ Master: {os.path.basename(MASTER_FILE_PATH)}")
    else:
        st.error("❌ Master NOT FOUND")
    master_upload = st.file_uploader("📤 Upload updated Master (optional)", type=["xlsx", "xls", "csv"], key="master_override")
    
    # Lulu Order Form
    if os.path.exists(LULU_ORDER_PATH):
        st.success(f"✅ Lulu: {os.path.basename(LULU_ORDER_PATH)}")
    else:
        st.warning("⚠️ Lulu Order Form missing")
    lulu_upload = st.file_uploader("📤 Upload updated Lulu Order (optional)", type=["xlsx", "xls", "csv"], key="lulu_override")
    
    # Nesto Order Form
    if os.path.exists(NESTO_ORDER_PATH):
        st.success(f"✅ Nesto: {os.path.basename(NESTO_ORDER_PATH)}")
    else:
        st.warning("⚠️ Nesto Order Form missing")
    nesto_upload = st.file_uploader("📤 Upload updated Nesto Order (optional)", type=["xlsx", "xls", "csv"], key="nesto_override")

# Determine which input to use
use_whsmith = bool(whsmith_text and whsmith_text.strip())

# Run the pipeline
if po_file is not None or use_whsmith:
    
    with st.spinner("Processing files and hunting for KR Codes..."):
        parsed_po_df = None
        
        if use_whsmith:
            # WHSmith email text route
            st.info("📧 Processing WHSmith email order text...")
            parsed_po_df = parse_whsmith_text(whsmith_text)
            if parsed_po_df is not None:
                st.success(f"✅ Parsed {len(parsed_po_df)} items from WHSmith email")
                
                # OTR→EA Conversion: use uploaded master if available, else hardcoded
                try:
                    if master_upload is not None:
                        master_upload.seek(0)
                        if master_upload.name.endswith('.csv'):
                            df_conv = pd.read_csv(master_upload, header=None)
                        else:
                            df_conv = pd.read_excel(master_upload, sheet_name='ZDET-PRICE', header=None)
                        master_upload.seek(0)  # Reset for apply_mappings later
                    else:
                        df_conv = pd.read_excel(MASTER_FILE_PATH, sheet_name='ZDET-PRICE', header=None)
                    
                    # Find header row
                    conv_header_idx = None
                    for ci, row in df_conv.iterrows():
                        row_str = " ".join([str(v).upper() for v in row if pd.notna(v)])
                        if "ARTICLE" in row_str and "GTIN" in row_str:
                            conv_header_idx = ci
                            break
                    
                    if conv_header_idx is not None:
                        df_conv.columns = df_conv.iloc[conv_header_idx]
                        df_conv = df_conv.iloc[conv_header_idx + 1:].reset_index(drop=True)
                        
                        # Build GTIN → conversion factor lookup
                        conv_col = 'Den. for Conversion 1'
                        if conv_col in df_conv.columns and 'GTIN' in df_conv.columns:
                            gtin_conv = {}
                            for _, row in df_conv.iterrows():
                                gtin = str(row.get('GTIN', '')).strip()
                                try:
                                    gtin = str(int(float(gtin)))
                                except (ValueError, TypeError):
                                    pass
                                conv = row.get(conv_col)
                                try:
                                    conv = float(conv)
                                except (ValueError, TypeError):
                                    conv = 1.0
                                if gtin and conv > 0:
                                    gtin_conv[gtin] = conv
                            
                            # Apply OTR→EA conversion
                            converted = 0
                            for idx, row in parsed_po_df.iterrows():
                                uom = str(row.get('UOM', '')).upper()
                                if uom == 'OTR':
                                    barcode = str(row.get('Barcode', '')).strip()
                                    try:
                                        barcode = str(int(float(barcode)))
                                    except (ValueError, TypeError):
                                        pass
                                    multiplier = gtin_conv.get(barcode, 1.0)
                                    orig_qty = row.get('Quantity', 0)
                                    parsed_po_df.at[idx, 'True Quantity'] = int(orig_qty * multiplier)
                                    parsed_po_df.at[idx, 'Conversion'] = f"{orig_qty}×{int(multiplier)}"
                                    converted += 1
                            
                            if converted > 0:
                                st.success(f"✅ Converted {converted} items: OTR → EA using Den. for Conversion 1")
                            else:
                                st.info("ℹ️ No OTR items found to convert (all quantities are already in EA)")
                        else:
                            st.warning(f"⚠️ Could not find '{conv_col}' column in Master File")
                    else:
                        st.warning("⚠️ Could not find header row in Master File for conversion")
                except Exception as e:
                    st.warning(f"⚠️ OTR conversion failed: {e}. Quantities kept as-is.")
            else:
                st.error("Could not parse any items from the pasted text. Check the format.")
        
        elif po_file is not None:
            # PDF/Excel PO route
            file_name = po_file.name.lower()
            if file_name.endswith(('.xlsx', '.xls')):
                st.info("Detected Excel PO file. Parsing as Carrefour fax conversion...")
                parsed_po_df = parse_carrefour_excel(po_file)
            else:
                parsed_po_df = process_pdf(po_file, api_key=ocr_api_key,
                                           ocr_engine=ocr_engine,
                                           ws_username=ws_username,
                                           ws_license=ws_license)
        
        if parsed_po_df is not None:
            # Normalize: ensure 'True Quantity' exists for all input types
            if 'True Quantity' not in parsed_po_df.columns and 'Quantity' in parsed_po_df.columns:
                parsed_po_df['True Quantity'] = parsed_po_df['Quantity']
            
            final_df = apply_mappings(parsed_po_df, 
                                       master_override=master_upload,
                                       lulu_override=lulu_upload,
                                       nesto_override=nesto_upload)
            
            if final_df is not None:
                # Anchor + auto-scroll to results
                st.markdown('<div id="results-anchor"></div>', unsafe_allow_html=True)
                st.subheader("📊 Final Output Table")
                st.dataframe(final_df, use_container_width=True)
                
                # Auto-scroll to the results table
                components.html("""
                    <script>
                        const el = window.parent.document.getElementById('results-anchor');
                        if (el) { el.scrollIntoView({ behavior: 'smooth', block: 'start' }); }
                    </script>
                """, height=0)
                
                # Prepare Downloads
                csv_buffer = final_df.to_csv(index=False).encode('utf-8')
                excel_buffer = io.BytesIO()
                with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                    final_df.to_excel(writer, index=False)
                
                dl_col1, dl_col2 = st.columns([1, 1])
                with dl_col1:
                    st.download_button(
                        label="📥 Download Final Excel",
                        data=excel_buffer.getvalue(),
                        file_name="Final_Mapped_Order.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                with dl_col2:
                    st.download_button(
                        label="📥 Download Final CSV",
                        data=csv_buffer,
                        file_name="Final_Mapped_Order.csv",
                        mime="text/csv"
                    )
                
                # --- COPY BUTTON: KR CODE + True Quantity ---
                st.divider()
                st.subheader("📋 Quick Copy: KR CODE + True Quantity")
                
                # Build copy-ready text (tab-separated for easy paste into Excel)
                copy_cols = ['KR CODE', 'True Quantity']
                if all(c in final_df.columns for c in copy_cols):
                    copy_df = final_df[copy_cols].copy()
                    # Format as tab-separated text
                    copy_text = "KR CODE\tTrue Quantity\n"
                    for _, row in copy_df.iterrows():
                        kr = str(row['KR CODE']) if pd.notna(row['KR CODE']) else ''
                        qty = str(row['True Quantity']) if pd.notna(row['True Quantity']) else ''
                        copy_text += f"{kr}\t{qty}\n"
                    
                    st.text_area(
                        "Select all → Copy (Ctrl+A, Ctrl+C)",
                        value=copy_text.strip(),
                        height=min(200, 40 + len(copy_df) * 20),
                        help="Tab-separated format — paste directly into Excel or Google Sheets"
                    )
                    
                    # Also offer as download for convenience
                    st.download_button(
                        label="📋 Download KR CODE + Qty Only",
                        data=copy_text.encode('utf-8'),
                        file_name="KR_CODE_Qty.tsv",
                        mime="text/tab-separated-values"
                    )
                else:
                    st.warning("KR CODE or True Quantity column not found in output.")
                
                # --- FLOATING "BACK TO TOP" BUTTON ---
                st.markdown("""
                    <style>
                    .back-to-top-btn {
                        position: fixed;
                        bottom: 30px;
                        right: 30px;
                        z-index: 9999;
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        color: white;
                        border: none;
                        border-radius: 50px;
                        padding: 14px 22px;
                        font-size: 16px;
                        font-weight: 600;
                        cursor: pointer;
                        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
                        transition: all 0.3s ease;
                        text-decoration: none;
                    }
                    .back-to-top-btn:hover {
                        transform: translateY(-2px);
                        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
                        color: white;
                    }
                    </style>
                    <a href="#top-anchor" class="back-to-top-btn" onclick="
                        var el = window.parent.document.getElementById('top-anchor');
                        if (el) { el.scrollIntoView({ behavior: 'smooth', block: 'start' }); }
                        return false;
                    ">⬆ Back to Top</a>
                """, unsafe_allow_html=True)

else:
    st.info("Upload a PO file (PDF/Excel) or paste WHSmith email text to get started.")