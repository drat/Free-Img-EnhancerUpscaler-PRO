"""
Image Enhancer Main App
-----------------------
Aplikasi utama Gradio untuk menjalankan UI image enhancer berbasis tab.
Untuk versi ini, hanya ada 1 tab "Enhance" yang memanggil demo dari modul app_enhance.

Created by _drat | 2025
"""

import gradio as gr

from app_enhance import create_demo as create_demo_enhance  # Import fungsi untuk membangun demo UI 'Enhance'
# from app_upscale import create_demo as create_demo_upscale  # Import fungsi untuk membangun demo UI 'Upscale'
from themes import IndonesiaTheme  # Impor tema custom

import warnings
warnings.filterwarnings("ignore")

# CSS untuk styling antarmuka
css = """
#col-left, #col-mid {
    margin: 0 auto;
    max-width: 400px;
    padding: 10px;
    border-radius: 15px;
    background-color: #f9f9f9;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
}
#col-right {
    margin: 0 auto;
    max-width: 400px;
    padding: 10px;
    border-radius: 15px;
    background: linear-gradient(180deg, #B6BBC4, #EEEEEE);
    color: white;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
}
#col-bott {
    margin: 0 auto;
    padding: 10px;
    border-radius: 15px;
    background-color: #f9f9f9;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
}
#banner {
    width: 100%;
    text-align: center;
    margin-bottom: 20px;
}
#run-button {
    background-color: #ff4b5c;
    color: white;
    font-weight: bold;
    padding: 30px;
    border-radius: 10px;
    cursor: pointer;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
}
#footer {
    text-align: center;
    margin-top: 20px;
    color: silver;
}
#markdown-silver {
    color: silver; /* Mengatur warna font Markdown menjadi silver */
}
"""


# Inisialisasi Gradio Blocks dengan custom CSS (style.css)
with gr.Blocks(css=css, theme=IndonesiaTheme()) as demo:
    with gr.Tabs():
        with gr.Tab(label="✨ Enhance"):
            create_demo_enhance()   # Panggil UI Enhance dari app_enhance.py (atau sesuai modul Anda)
        # with gr.Tab(label="🚀 Upscale"):
        #     create_demo_upscale()   # Panggil UI Upscale dari app_upscale.py (atau sesuai modul Anda)

# Jalankan aplikasi Gradio (local/web)
demo.queue(api_open=False).launch(show_api=False)
