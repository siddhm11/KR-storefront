"""Full duplicate barcode analysis: Master File + Lulu + Nesto Order Forms."""
import pandas as pd
import sys
import io

# Force UTF-8 output to file
out = io.open("dupe_report.txt", "w", encoding="utf-8")

def clean_key(val):
    if pd.isna(val):
        return ""
    try:
        return str(int(float(val)))
    except (ValueError, TypeError):
        return str(val).strip()


# ============================================================
# 1. MASTER FILE (ZDET-PRICE)
# ============================================================
out.write("=" * 80 + "\n")
out.write("MASTER FILE - Bel Article Master 12.25.xlsx (ZDET-PRICE)\n")
out.write("=" * 80 + "\n")

df = pd.read_excel("Bel Article Master 12.25.xlsx", sheet_name="ZDET-PRICE", header=None)

header_idx = None
for idx, row in df.iterrows():
    if idx > 50:
        break
    row_str = " ".join([str(v).upper() for v in row if pd.notna(v)])
    if "ARTICLE" in row_str and "GTIN" in row_str:
        header_idx = idx
        break

df.columns = df.iloc[header_idx]
df = df.iloc[header_idx + 1:].reset_index(drop=True)
df = df.loc[:, ~df.columns.duplicated()]

df["GTIN_Clean"] = df["GTIN"].apply(clean_key)
df_valid = df[df["GTIN_Clean"] != ""].copy()

desc_col = "Material Description" if "Material Description" in df.columns else "Description"

price_col = None
for c in df.columns:
    cu = str(c).upper()
    if "SELLING" in cu and "PRICE" in cu:
        price_col = c
        break
if price_col is None:
    for c in df.columns:
        cu = str(c).upper()
        if "NET" in cu and "PRICE" in cu:
            price_col = c
            break
if price_col is None:
    for c in df.columns:
        cu = str(c).upper()
        if "PRICE" in cu:
            price_col = c
            break

out.write(f"Total rows: {len(df)}\n")
out.write(f"Rows with valid GTIN: {len(df_valid)}\n")
out.write(f"Unique GTINs: {df_valid['GTIN_Clean'].nunique()}\n")
if price_col:
    out.write(f"Price column: {price_col}\n")

dupes = df_valid[df_valid.duplicated(subset="GTIN_Clean", keep=False)].sort_values("GTIN_Clean")
out.write(f"Duplicate GTINs: {dupes['GTIN_Clean'].nunique()}\n")
out.write(f"Total rows involved: {len(dupes)}\n\n")

if len(dupes) > 0:
    count = 0
    for gtin, group in dupes.groupby("GTIN_Clean"):
        count += 1
        out.write(f"--- #{count}: GTIN {gtin} ({len(group)} entries) ---\n")
        for _, row in group.iterrows():
            article = str(row.get("Article", "?"))
            desc = str(row.get(desc_col, ""))[:60]
            price = row.get(price_col, "N/A") if price_col else "N/A"
            try:
                price_str = f"{float(price):.3f}"
            except (ValueError, TypeError):
                price_str = str(price)
            out.write(f"    Article: {article:<10}  Price: {price_str:<10}  Desc: {desc}\n")
        out.write("\n")

    # Summary
    conflict_count = 0
    same_count = 0
    for gtin, group in dupes.groupby("GTIN_Clean"):
        if group["Article"].nunique() > 1:
            conflict_count += 1
        else:
            same_count += 1
    out.write(f"CONFLICTS (same GTIN, different Article): {conflict_count}\n")
    out.write(f"REDUNDANT (same GTIN, same Article):      {same_count}\n")


# ============================================================
# 2. ORDER FORMS
# ============================================================
def analyze_order_form(filepath, label):
    out.write("\n\n" + "=" * 80 + "\n")
    out.write(f"{label} - {filepath}\n")
    out.write("=" * 80 + "\n")
    
    try:
        df_dict = pd.read_excel(filepath, sheet_name=None, header=None)
    except FileNotFoundError:
        out.write(f"  FILE NOT FOUND\n")
        return
    
    df_order = None
    for sheet_name, df_sheet in df_dict.items():
        for idx, row in df_sheet.iterrows():
            if idx > 100:
                break
            row_str = " ".join([str(v).upper() for v in row if pd.notna(v)])
            if ("BARCODES" in row_str or "BARCODE" in row_str) and ("KR CODE" in row_str or "SAP CODE" in row_str):
                df_sheet.columns = df_sheet.iloc[idx]
                df_order = df_sheet.iloc[idx + 1:].reset_index(drop=True)
                break
        if df_order is not None:
            break
    
    if df_order is None:
        out.write("  Could not find header row\n")
        return
    
    df_order = df_order.loc[:, ~df_order.columns.duplicated()]
    df_order.columns = [str(c).strip().upper() for c in df_order.columns]
    
    bc_col = None
    for c in df_order.columns:
        if "BARCODE" in c:
            bc_col = c
            break
    
    kr_col = "KR CODE" if "KR CODE" in df_order.columns else ("SAP CODE" if "SAP CODE" in df_order.columns else None)
    desc_col = "PRODUCT" if "PRODUCT" in df_order.columns else ("PRODUCT DESCRIPTION" if "PRODUCT DESCRIPTION" in df_order.columns else None)
    
    if not bc_col:
        out.write("  No barcode column found\n")
        return
    
    df_order["BC_Clean"] = df_order[bc_col].apply(clean_key)
    df_valid = df_order[df_order["BC_Clean"] != ""].copy()
    
    out.write(f"Barcode column: {bc_col}\n")
    out.write(f"Total rows: {len(df_order)}\n")
    out.write(f"Rows with valid barcode: {len(df_valid)}\n")
    out.write(f"Unique barcodes: {df_valid['BC_Clean'].nunique()}\n")
    
    dupes = df_valid[df_valid.duplicated(subset="BC_Clean", keep=False)].sort_values("BC_Clean")
    out.write(f"Duplicate barcodes: {dupes['BC_Clean'].nunique()}\n")
    out.write(f"Total rows involved: {len(dupes)}\n\n")
    
    if len(dupes) > 0:
        count = 0
        for bc, group in dupes.groupby("BC_Clean"):
            count += 1
            out.write(f"--- #{count}: Barcode {bc} ({len(group)} entries) ---\n")
            for _, row in group.iterrows():
                kr = str(row.get(kr_col, "?")) if kr_col else "?"
                desc = str(row.get(desc_col, ""))[:60] if desc_col else ""
                out.write(f"    KR Code: {kr:<10}  Desc: {desc}\n")
            out.write("\n")
    else:
        out.write("No duplicates found!\n")


analyze_order_form("Lulu Order Form 8.24 (1).xlsx", "LULU ORDER FORM")
analyze_order_form("Nesto Order Form 8.24.xlsx", "NESTO ORDER FORM")

out.close()
print("Report written to dupe_report.txt")
