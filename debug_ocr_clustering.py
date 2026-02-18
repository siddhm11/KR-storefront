"""Debug OCR clustering logic (Safe Encoding)."""
import json

with open("ocr_api_result.json", "r", encoding="utf-8") as f:
    data = json.load(f)

lines = data['ParsedResults'][0]['TextOverlay']['Lines']

def get_left(line): return line['Words'][0]['Left']

sorted_lines = sorted(lines, key=get_left, reverse=True)

current_group = []
current_left = -1
TOLERANCE = 20.0 # Strict tolerance might be the issue?

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
if current_group: groups.append(current_group)

# Write output to file safely
with open("debug_clustering_output.txt", "w", encoding="utf-8") as out:
    out.write(f"found {len(groups)} raw groups.\n\n")

    valid_rows = 0
    dropped_rows = 0

    for i, group in enumerate(groups):
        row_text = []
        has_barcode = False
        
        for line in group:
            text = line['LineText']
            top = line['Words'][0]['Top']
            row_text.append(f"[{int(top)}: {text}]")
            
            # Loose check for barcode (100-400 Top range + digits)
            if 100 <= top <= 400:
                bc = text.replace(" ", "").strip()
                if any(c.isdigit() for c in bc) and len(bc) > 5:
                    has_barcode = True
                    
        if has_barcode:
            valid_rows += 1
            out.write(f"✅ Row {i} (Left ~{group[0]['Words'][0]['Left']:.0f}): {' '.join(row_text)}\n")
        else:
            dropped_rows += 1
            out.write(f"❌ Dropped Row {i} (Left ~{group[0]['Words'][0]['Left']:.0f}): {' '.join(row_text)}\n")

    out.write(f"\nSummary: {valid_rows} Valid, {dropped_rows} Dropped.")
