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

Ensure `requirements.txt` is present. Vercel supports Python runtimes, but for Streamlit specifically, consider **Streamlit Community Cloud** for best compatibility.

If deploying to Vercel, you may need to configure a `vercel.json` or use a custom build command.

## Files

- `app.py`: Main application logic.
- `lulu.py`: Legacy/alternative script.
- `requirements.txt`: Python dependencies.
