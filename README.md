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

**Crucial Step**: In your Vercel Project Settings > Build & Development settings:
1. **Framework Preset**: select "Other"
2. **Build Command**: `mkdir public` (Streamlit doesn't need a build step, but Vercel expects one)
3. **Install Command**: `pip install -r requirements.txt --break-system-packages`
   - *Note: Vercel uses a system-managed Python environment, so `--break-system-packages` is required to install libraries globally in the container.*
4. **Output Directory**: `public`

⚠️ **Warning**: Vercel has a 10-second timeout for serverless functions on the free tier. Streamlit apps are long-running processes and WILL likely timeout or fail to run properly on Vercel. **Streamlit Community Cloud is highly recommended instead.**

## Files

- `app.py`: Main application logic.
- `lulu.py`: Legacy/alternative script.
- `requirements.txt`: Python dependencies.
