import streamlit as st
import pdfplumber
import pandas as pd
import numpy as np
import re
import io
import requests
from ocr_lib import get_carrefour_data
import requests # Ensure requests is imported
from ocr_lib import get_carrefour_data # Import OCR logic

# Set up the web page
st.set_page_config(page_title="Universal PO & Multi-Layer Mapper", layout="wide")
st.title("📄 Universal PO Converter & Multi-Layer Mapper")
st.markdown("Works for **LuLu, Nesto, & Carrefour (Fax)**! Upload your **Master File**, the **Retailer Order Form** (Fallback), and the **PDF PO**. The app dynamically adapts to different column names to find the exact KR CODE!")

# API Key for OCR (Optional)
with st.sidebar:
    st.header("OCR Settings")
    ocr_api_key = st.text_input("OCR.space API Key", value="helloworld", type="password", help="Get a free key from https://ocr.space/ocrapi")
    st.info("Default 'helloworld' key works for testing but is rate-limited.")

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

# --- FUNCTION 3: PROCESS PDF AND CALCULATE TRUE QUANTITY ---
def process_pdf(pdf_file, api_key=None):
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
            if api_key:
                df_ocr = get_carrefour_data(pdf_file, api_key=api_key)
                if df_ocr is not None and not df_ocr.empty:
                    st.success(f"OCR Successful! Extracted {len(df_ocr)} rows.")
                    return df_ocr
                else:
                    st.error("OCR failed to extract data. Is the API Key valid?")
                    return None
            else:
                st.error("Text not found and no OCR API Key provided.")
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
                
            master_gtin_dict = df_master[df_master['GTIN'] != ""].set_index('GTIN')['Article'].to_dict()
            master_desc_dict = df_master[df_master[desc_col_name] != ""].set_index(desc_col_name)['Article'].to_dict() if desc_col_name in df_master.columns else {}
            st.success("✅ Master Mapping (ZDET-PRICE) loaded!")
        else:
            st.error("🚨 Missing GTIN/Article headers in Master File.")
            return None

        # ==========================================
        # 2. LOAD ORDER FORM (FALLBACK)
        # ==========================================
        order_barcode_dict, order_retailer_dict, order_desc_dict = {}, {}, {}
        
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
                if of_barcode and of_kr:
                    df_order[of_barcode] = df_order[of_barcode].apply(clean_key)
                    order_barcode_dict = df_order[df_order[of_barcode] != ""].set_index(of_barcode)[of_kr].to_dict()
                
                if of_retailer and of_kr:
                    df_order[of_retailer] = df_order[of_retailer].apply(clean_key)
                    order_retailer_dict = df_order[df_order[of_retailer] != ""].set_index(of_retailer)[of_kr].to_dict()
                    
                if of_desc and of_kr:
                    df_order[of_desc] = df_order[of_desc].apply(clean_desc)
                    order_desc_dict = df_order[df_order[of_desc] != ""].set_index(of_desc)[of_kr].to_dict()
                    
                st.success(f"✅ Order Form Fallback loaded! (Found: {of_retailer or 'N/A'} & {of_kr or 'N/A'})")
            else:
                st.warning("⚠️ Could not find Barcode & KR/SAP Code headers in the Order Form. Skipping fallback.")

        # ==========================================
        # 3. APPLY MULTI-LAYER MAPPING TO PO
        # ==========================================
        # Dynamically identify PO description column (LuLu vs Nesto)
        po_desc_col = 'Article Description + Add.Info' if 'Article Description + Add.Info' in df_po.columns else ('Description' if 'Description' in df_po.columns else None)
        
        # Clean PO matching keys safely
        if 'Barcode' in df_po.columns: df_po['Barcode_Clean'] = df_po['Barcode'].apply(clean_key)
        else: df_po['Barcode_Clean'] = ""
            
        if po_desc_col: df_po['Clean PO Desc'] = df_po[po_desc_col].apply(clean_desc)
        else: df_po['Clean PO Desc'] = ""
            
        if 'Article' in df_po.columns: df_po['Article Code'] = df_po['Article'].apply(clean_key)
        else: df_po['Article Code'] = ""

        # The Waterfall Mapping Function
        def find_kr_code(row):
            barcode = row.get('Barcode_Clean', "")
            po_article = row.get('Article Code', "")
            desc = row.get('Clean PO Desc', "")
            
            # Layer 1: Master File Barcode (GTIN)
            if barcode != "" and barcode in master_gtin_dict:
                return master_gtin_dict[barcode]
                
            # Layer 2: Order Form Barcode
            if barcode != "" and barcode in order_barcode_dict:
                return order_barcode_dict[barcode]
                
            # Layer 3: Order Form Retailer Code (LuLu Code / Nesto Code → PO Article)
            if po_article != "" and po_article in order_retailer_dict:
                return order_retailer_dict[po_article]
                
            # Layer 4: Master File Description
            if desc != "" and desc in master_desc_dict:
                return master_desc_dict[desc]
                
            # Layer 5: Order Form Description
            if desc != "" and desc in order_desc_dict:
                return order_desc_dict[desc]
                
            return "Not Found"

        df_po['KR CODE'] = df_po.apply(find_kr_code, axis=1)

        # Output correct columns based on file type
        desired_columns = ['KR CODE', 'True Quantity', 'Barcode', 'Article']
        if po_desc_col: desired_columns.append(po_desc_col)
        
        final_cols = [col for col in desired_columns if col in df_po.columns]
        return df_po[final_cols]
        
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
    st.subheader("3. PDF PO")
    po_file = st.file_uploader("Upload LuLu/Nesto PDF PO", type=["pdf"])

st.divider()

# Run the pipeline only when Master and PO are provided (Order Form is an optional bonus)
if po_file is not None and master_file is not None:
    
    with st.spinner("Processing files and hunting for KR Codes..."):
        parsed_po_df = process_pdf(po_file, api_key=ocr_api_key)
        
        if parsed_po_df is not None:
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