import streamlit as st
import pdfplumber
import pandas as pd
import numpy as np
import re
import io
import requests
from difflib import SequenceMatcher
from ocr_lib import get_carrefour_data, parse_carrefour_excel

# Set up the web page
st.set_page_config(page_title="Universal PO & Multi-Layer Mapper", layout="wide")
st.title("📄 Universal PO Converter & Multi-Layer Mapper")
st.markdown("Works for **LuLu, Nesto, & Carrefour (Fax)**! Upload your **Master File**, the **Retailer Order Form** (Fallback), and the **PDF PO**. The app dynamically adapts to different column names to find the exact KR CODE!")

# OCR Settings (for scanned PDFs like Carrefour Fax)
with st.sidebar:
    st.header("OCR Settings (Carrefour Fax)")
    ocr_engine = st.selectbox("OCR Engine", ["ocrspace", "webservice"], index=0,
        help="'webservice' = onlineocr.net engine (better quality, needs signup). 'ocrspace' = free fallback.")
    
    if ocr_engine == 'webservice':
        st.markdown("[Sign up free at ocrwebservice.com](https://www.ocrwebservice.com/account/signup) — 25 pages/day")
        ws_username = st.text_input("Username", help="Your ocrwebservice.com username")
        ws_license = st.text_input("License Code", type="password", help="Your ocrwebservice.com license code")
        ocr_api_key = None
    else:
        ocr_api_key = st.text_input("OCR.space API Key", value="helloworld", type="password", help="Get a free key from https://ocr.space/ocrapi")
        ws_username = None
        ws_license = None
        st.info("Default 'helloworld' key works for testing but is rate-limited.")
    
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
            row_str = " ".join([str(val).upper() for val in row if val is not None])
            if ("QUANTITY" in row_str or "ORD.QTY" in row_str) and ("UOM" in row_str or "UNIT" in row_str):
                header_idx = i
                break
        
        if header_idx is None:
            # Fallback: pad at end and return raw
            max_cols = max(len(r) for r in all_rows)
            all_rows = [r + [None] * (max_cols - len(r)) for r in all_rows]
            df = pd.DataFrame(all_rows)
            df = df.map(lambda x: str(x).replace('\n', ' ').strip() if pd.notnull(x) else x)
            st.warning("Could not automatically find the Quantity/UOM headers. Returning raw data.")
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
        df = df.iloc[header_idx + 1:].reset_index(drop=True)
        
        first_col_name = df.columns[0]
        if pd.notnull(first_col_name):
            df = df[df[first_col_name] != first_col_name]
            
        df = df.dropna(how='all')
        df = df.loc[:, df.columns.notna()]

        # DYNAMIC TRUE QUANTITY CALCULATION (Adapts to LuLu & Nesto columns)
        def calculate_true_quantity(row):
            try:
                # Dynamically fetch Quantity (LuLu = 'Quantity', Nesto = 'Ord.Qty')
                qty_val = row.get('Quantity') if 'Quantity' in row.index else row.get('Ord.Qty', 0)
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
        return df

    except Exception as e:
        st.error(f"An unexpected error occurred while processing the PDF: {e}")
        return None

# --- FUNCTION 4: APPLY MULTI-LAYER MAPPING ---
def apply_mappings(df_po, master_file, order_form_file):
    try:
        # ==========================================
        # 1. LOAD MASTER FILE (ZDET-PRICE)
        # ==========================================
        if master_file.name.endswith('.csv'):
            df_master_raw = pd.read_csv(master_file, header=None)
        else:
            try:
                df_master_raw = pd.read_excel(master_file, sheet_name='ZDET-PRICE', header=None)
            except ValueError:
                st.error("🚨 Could not find a sheet named 'ZDET-PRICE' in the Master File.")
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
        # 2. LOAD ORDER FORM (FALLBACK)
        # ==========================================
        order_barcode_dict, order_retailer_dict, order_desc_dict = {}, {}, {}
        order_fuzzy_list = []
        
        if order_form_file is not None:
            if order_form_file.name.endswith('.csv'):
                df_order_dict = {"CSV": pd.read_csv(order_form_file, header=None)}
            else:
                df_order_dict = pd.read_excel(order_form_file, sheet_name=None, header=None)
            
            df_order = None
            # Scan all sheets for the Order Form table
            for sheet_name, df_sheet in df_order_dict.items():
                for idx, row in df_sheet.iterrows():
                    if idx > 100: break
                    row_str = " ".join([str(val).upper() for val in row.values if pd.notnull(val)])
                    # Match LuLu (BARCODES + KR CODE) OR Nesto (BARCODE + SAP CODE)
                    if ("BARCODES" in row_str or "BARCODE" in row_str) and ("KR CODE" in row_str or "SAP CODE" in row_str):
                        df_sheet.columns = df_sheet.iloc[idx]
                        df_order = df_sheet.iloc[idx + 1:].reset_index(drop=True)
                        break
                if df_order is not None: break
                
            if df_order is not None:
                df_order = df_order.loc[:, ~df_order.columns.duplicated()]
                df_order.columns = [str(c).strip().upper() for c in df_order.columns]
                
                # DYNAMIC COLUMN IDENTIFICATION (works for both LuLu and Nesto)
                of_barcode = 'BARCODES' if 'BARCODES' in df_order.columns else ('BARCODE (PER UNIT)' if 'BARCODE (PER UNIT)' in df_order.columns else None)
                of_kr = 'KR CODE' if 'KR CODE' in df_order.columns else ('SAP CODE' if 'SAP CODE' in df_order.columns else None)
                of_retailer = 'LULU CODE' if 'LULU CODE' in df_order.columns else ('NESTO CODE' if 'NESTO CODE' in df_order.columns else None)
                of_desc = 'PRODUCT' if 'PRODUCT' in df_order.columns else ('PRODUCT DESCRIPTION' if 'PRODUCT DESCRIPTION' in df_order.columns else None)
                
                # Build Order Form Dictionaries based on what was found
                # Store (kr_code, source_row) tuples for provenance tracking
                if of_barcode and of_kr:
                    df_order[of_barcode] = df_order[of_barcode].apply(clean_key)
                    order_barcode_dict = {row[of_barcode]: (row[of_kr], idx + 2)
                                          for idx, row in df_order.iterrows()
                                          if row[of_barcode] != ""}
                
                if of_retailer and of_kr:
                    df_order[of_retailer] = df_order[of_retailer].apply(clean_key)
                    order_retailer_dict = {row[of_retailer]: (row[of_kr], idx + 2)
                                           for idx, row in df_order.iterrows()
                                           if row[of_retailer] != ""}
                    
                if of_desc and of_kr:
                    df_order[of_desc] = df_order[of_desc].apply(clean_desc)
                    order_desc_dict = {row[of_desc]: (row[of_kr], idx + 2)
                                       for idx, row in df_order.iterrows()
                                       if row[of_desc] != ""}
                    # Build order form fuzzy list
                    for idx, row in df_order.iterrows():
                        desc_val = str(row.get(of_desc, '')).strip()
                        if desc_val and desc_val.upper() != 'NAN':
                            order_fuzzy_list.append({
                                'desc': clean_desc_fuzzy(desc_val),
                                'article': row.get(of_kr, ''),
                                'row': idx + 2,
                                'price': None,
                                'raw_desc': desc_val
                            })
                    
                st.success(f"✅ Order Form Fallback loaded! ({len(order_fuzzy_list)} items for fuzzy match)")
            else:
                st.warning("⚠️ Could not find Barcode & KR/SAP Code headers in the Order Form. Skipping fallback.")

        # ==========================================
        # 3. APPLY MULTI-LAYER MAPPING TO PO
        # ==========================================
        # Dynamically identify PO description column (LuLu vs Nesto)
        po_desc_col = 'Article Description + Add.Info' if 'Article Description + Add.Info' in df_po.columns else ('Description' if 'Description' in df_po.columns else None)
        
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
        if 'Barcode' in df_po.columns: df_po['Barcode_Clean'] = df_po['Barcode'].apply(clean_key)
        else: df_po['Barcode_Clean'] = ""
            
        if po_desc_col: df_po['Clean PO Desc'] = df_po[po_desc_col].apply(clean_desc)
        else: df_po['Clean PO Desc'] = ""
        
        # Also build fuzzy-friendly PO description
        if po_desc_col: df_po['Fuzzy PO Desc'] = df_po[po_desc_col].apply(clean_desc_fuzzy)
        else: df_po['Fuzzy PO Desc'] = ""
            
        if 'Article' in df_po.columns: df_po['Article Code'] = df_po['Article'].apply(clean_key)
        else: df_po['Article Code'] = ""
        
        # Extract PO price for each row
        if po_price_col:
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
                    
                # Layer 2: Order Form Barcode
                if barcode != "" and barcode in order_barcode_dict:
                    kr, src_row = order_barcode_dict[barcode]
                    return pd.Series([kr, "L2: Order (Barcode)", barcode, src_row])
                    
                # Layer 3: Order Form Retailer Code
                if po_article != "" and po_article in order_retailer_dict:
                    kr, src_row = order_retailer_dict[po_article]
                    return pd.Series([kr, "L3: Order (Retailer)", po_article, src_row])
                    
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
        desired_columns = ['KR CODE', 'True Quantity', 'Match Source', 'Match Key', 'Source Row', 'FAM', 'Barcode', 'Article']
        if po_desc_col: desired_columns.append(po_desc_col)
        
        final_cols = [col for col in desired_columns if col in df_po.columns]
        result = df_po[final_cols].reset_index(drop=True)
        result.index = result.index + 1  # Start row numbers from 1
        return result
        
    except Exception as e:
        st.error(f"An error occurred while applying mappings: {e}")
        return None

# --- UI LAYOUT ---
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("1. Master Mapping")
    master_file = st.file_uploader("Upload 'Bel Article' File", type=["xlsx", "xls", "csv"])

with col2:
    st.subheader("2. Order Form (Fallback)")
    order_file = st.file_uploader("Upload Retailer Order File (Optional)", type=["xlsx", "xls", "csv"])

with col3:
    st.subheader("3. PO File")
    po_file = st.file_uploader("Upload PO (PDF or Excel from OCR)", type=["pdf", "xlsx", "xls"])

st.divider()

# Run the pipeline only when Master and PO are provided (Order Form is an optional bonus)
if po_file is not None and master_file is not None:
    
    with st.spinner("Processing files and hunting for KR Codes..."):
        # Route: Excel PO (from onlineocr.net) vs PDF PO
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
            # LuLu/Nesto have it from calculate_true_quantity(); OCR/Excel only have 'Quantity'
            if 'True Quantity' not in parsed_po_df.columns and 'Quantity' in parsed_po_df.columns:
                parsed_po_df['True Quantity'] = parsed_po_df['Quantity']
            
            final_df = apply_mappings(parsed_po_df, master_file, order_file)
            
            if final_df is not None:
                st.subheader("📊 Final Output Table")
                st.dataframe(final_df, use_container_width=True)
                
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

elif po_file is None or master_file is None:
    st.info("Waiting for the Master File and PDF PO to be uploaded...")