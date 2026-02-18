# KR Storefront - Universal PO Mapper

This application automates the mapping of Purchase Orders (POs) from LuLu and Nesto to internal Khimji Ramdas (KR) codes.

## Features

- **Universal Parser**: Handles both LuLu and Nesto PDF formats automatically.
- **Dynamic Mapping**: Supports fallback to Order Forms if Master File lookup fails.
- **Smart Column Alignment**: Fixes PDF parsing issues where columns are dropped on later pages.
- **User Friendly**: Simple drag-and-drop interface powered by Streamlit.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/siddhm11/KR-storefront.git
   cd KR-storefront
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   # Windows
   .\venv\Scripts\activate
   # Mac/Linux
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the app locally:
```bash
streamlit run app.py
```

## Deployment (Vercel)

### 🚨 ERROR: "streamlit: command not found"?
This error happens if you set **Build Command** to `streamlit run app.py`.
**FIX:** Change **Build Command** to `mkdir -p public` (or leave empty).

### 🚨 ERROR: "Bundle size exceeds limit"?
Vercel has a hard 250MB limit which Streamlit + Pandas often exceeds.

### 🏆 RECOMMENDED: Streamlit Community Cloud
Since Vercel is not designed for Streamlit (it will timeout after 10s), the **best** loading experience is:
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Connect GitHub
3. Select this repo: `siddhm11/KR-storefront`
4. Click **Deploy**. Done!

## Files

- `app.py`: Main application logic.
- `lulu.py`: Legacy/alternative script.
- `requirements.txt`: Python dependencies.
