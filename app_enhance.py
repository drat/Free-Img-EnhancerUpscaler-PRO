"""
Image Enhancer App
------------------
Aplikasi berbasis Gradio untuk meningkatkan kualitas gambar dengan berbagai teknik pemrosesan citra
seperti perbaikan resolusi, penajaman, dan penghilangan noise. 
Dapat digunakan secara lokal maupun sebagai web-app AI image enhancement.

Created by _drat | 2025
"""

# Import library standar dan eksternal yang dibutuhkan
import os
import subprocess
import spaces  # Library dari HuggingFace untuk GPU management & dekorator
import torch   # Library untuk deep learning, cek GPU
import cv2     # OpenCV untuk pengolahan gambar
import uuid    # Untuk generate nama file unik
import gradio as gr # Untuk membuat UI web berbasis Python
import numpy as np  # Operasi numerik, termasuk array

from PIL import Image   # Untuk memproses dan menyimpan gambar
# Import arsitektur model dan utilitas untuk enhancement & upscaling
from basicsr.archs.srvgg_arch import SRVGGNetCompact
from gfpgan.utils import GFPGANer
from realesrgan.utils import RealESRGANer

import warnings
warnings.filterwarnings("ignore")

# Fungsi untuk menjalankan command shell (misal: wget file model)
def runcmd(cmd, verbose = False):
    process = subprocess.Popen(
        cmd,
        stdout = subprocess.PIPE,
        stderr = subprocess.PIPE,
        text = True,
        shell = True
    )
    std_out, std_err = process.communicate()
    if verbose:
        print(std_out.strip(), std_err)
    pass

# Pastikan model AI (GFPGAN dan RealESRGAN) sudah terdownload. Jika belum, otomatis download.
if not os.path.exists('GFPGANv1.4.pth'):
    runcmd("wget https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth -P .")
if not os.path.exists('realesr-general-x4v3.pth'):
    runcmd("wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth -P .")

# Inisialisasi model Real-ESRGAN untuk upscaling gambar
model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
model_path = 'realesr-general-x4v3.pth'
half = True if torch.cuda.is_available() else False  # Pakai mode half-precision jika ada GPU (lebih cepat)
upsampler = RealESRGANer(
    scale=4, model_path=model_path, model=model, 
    tile=0, tile_pad=10, pre_pad=0, half=half
)

# Fungsi utama untuk enhance gambar (dipanggil saat tombol di UI ditekan)
@spaces.GPU(duration=15)
def enhance_image(
    input_image: Image,      # Gambar input dari user (PIL Image)
    scale: int,              # Skala upscaling (misal: 2x, 4x)
    enhance_mode: str,       # Mode enhance: face saja, image saja, atau keduanya
):
    only_face = enhance_mode == "Only Face Enhance"
    
    # Pilih mode enhancer: hanya wajah, hanya gambar, atau kombinasi
    if enhance_mode == "Only Face Enhance":
        face_enhancer = GFPGANer(
            model_path='GFPGANv1.4.pth', upscale=scale, 
            arch='clean', channel_multiplier=2
        )
    elif enhance_mode == "Only Image Enhance":
        face_enhancer = None  # Tidak enhance wajah, hanya upscaling gambar saja
    else:
        face_enhancer = GFPGANer(
            model_path='GFPGANv1.4.pth', upscale=scale, 
            arch='clean', channel_multiplier=2, bg_upsampler=upsampler
        )
    
    # Konversi gambar input ke format BGR (OpenCV)
    img = cv2.cvtColor(np.array(input_image), cv2.COLOR_RGB2BGR)
    h, w = img.shape[0:2]

    # Optional: perbesar gambar jika terlalu kecil (kurang dari 300px)
    if h < 300:
        img = cv2.resize(img, (w * 2, h * 2), interpolation=cv2.INTER_LANCZOS4)
    
    # Proses enhance gambar (tergantung mode)
    if face_enhancer is not None:
        # Enhance wajah (GFPGAN), output gambar hasil enhancement
        _, _, output = face_enhancer.enhance(
            img, has_aligned=False, only_center_face=only_face, paste_back=True
        )
    else:
        # Hanya upscaling image (tanpa enhance wajah)
        output, _ = upsampler.enhance(img, outscale=scale)

    # --- Optional resize (tidak digunakan)
    # if scale != 2:
    #     interpolation = cv2.INTER_AREA if scale < 2 else cv2.INTER_LANCZOS4
    #     h, w = img.shape[0:2]
    #     output = cv2.resize(output, (int(w * scale / 2), int(h * scale / 2)), interpolation=interpolation)
    
    # Batasi hasil enhance agar tidak lebih dari 3480px (biar aman dan efisien)
    h, w = output.shape[0:2]
    max_size = 3480
    if h > max_size:
        w = int(w * max_size / h)
        h = max_size
    if w > max_size:
        h = int(h * max_size / w)
        w = max_size
    
    # Resize hasil akhir
    output = cv2.resize(output, (w, h), interpolation=cv2.INTER_LANCZOS4)

    # Konversi kembali ke RGB & PIL Image (untuk output Gradio)
    enhanced_image = Image.fromarray(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
    
    # Simpan gambar hasil enhance ke folder output dengan nama unik
    tmpPrefix = "/tmp/gradio/"
    extension = 'png'
    # targetDir = f"{tmpPrefix}output/"
    targetDir = "output/"
    if not os.path.exists(targetDir):
        os.makedirs(targetDir)

    enhanced_path = f"{targetDir}{uuid.uuid4()}.{extension}"
    enhanced_image.save(enhanced_path, quality=100)
        
    return enhanced_image, enhanced_path  # Return: image untuk preview dan path file untuk download


#
# DEMO DI SINI
#

def create_demo() -> gr.Blocks:
    with gr.Blocks() as demo:
        # Feature Image dengan border subtle & shadow
        gr.HTML(
            """
            <div style='display:flex; justify-content:center;'>
                <img src='https://i.ibb.co/jkkT7MZQ/feature-image-enhancer.jpg' 
                     alt='Feature Image' 
                     style='height:200px; width:auto; border-radius:24px; box-shadow:0 4px 18px #ffa07430; border:2.5px solid #ffe0b3;' 
                     id='feature-image'/>
            </div>
            """
        )
        # Judul & Tagline dengan emoji & gradient text
        gr.Markdown("""
            <div style="text-align:center; margin-top: -10px; margin-bottom: 14px;">
                <span style="font-size:2.1rem; font-family:'Quicksand',sans-serif; font-weight:800; background: linear-gradient(90deg,#d84040,#ff914d 60%); -webkit-background-clip: text; color:transparent; display:inline-block;">
                    <span style="vertical-align:middle;">📸</span> Image Enhancer <b>Pro</b> <span style="font-size:1.1em;vertical-align:middle;">✨</span>
                </span><br>
                <span style="font-size:1.12rem; color:#d84040; font-family:'Inter',sans-serif;">
                    <span style="vertical-align:middle;">🤖</span> Foto blur? Jadikan jernih & tajam pakai AI! 
                </span>
            </div>
        """)

        # Input dan Output dalam card, dengan icon pada title
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Group(elem_id="input-card"):
                    gr.Markdown("<div class='card-title'><span style='font-size:1.3em;vertical-align:middle;'>🖼️</span> <b>Gambar Asli</b></div>")
                    input_image = gr.Image(label="", type="pil", elem_id="input-image", show_label=False)
                    gr.Markdown("<div class='hint'><span style='font-size:1.1em;'>⬆️</span> Upload foto format <b>JPG/PNG</b>, max 5MB</div>")
            with gr.Column(scale=1):
                with gr.Group(elem_id="output-card"):
                    gr.Markdown("<div class='card-title'><span style='font-size:1.3em;vertical-align:middle;'>🌈</span> <b>Hasil AI Enhance</b></div>")
                    output_image = gr.Image(label="", type="pil", interactive=False, elem_id="output-image", show_label=False)
                    enhance_image_path = gr.File(label="⬇️ Download (.png)", interactive=False, elem_id="download-btn")
                    gr.Markdown("""
                        <div class='hint' style="color:#ff914d;"><span style='font-size:1.2em;'>💡</span> Klik untuk download hasilnya!</div>
                    """)

        # Kontrol dengan ikon pada label
        with gr.Group(elem_id="control-card"):
            with gr.Row():
                scale = gr.Slider(
                    minimum=1,
                    maximum=4,
                    value=2,
                    step=1,
                    label="🔎 Upscale x",
                    elem_id="scale-slider"
                )
                enhance_mode = gr.Dropdown(
                    label="🛠️ Mode Enhance",
                    choices=[
                        "Only Face Enhance",
                        "Only Image Enhance",
                        "Face Enhance + Image Enhance",
                    ],
                    value="Face Enhance + Image Enhance",
                    elem_id="mode-dropdown"
                )
                g_btn = gr.Button("🚀 Enhance Sekarang!", elem_id="enhance-btn", size="lg")

        g_btn.click(
            fn=enhance_image,
            inputs=[input_image, scale, enhance_mode],
            outputs=[output_image, enhance_image_path],
        )
        # Footer dengan ikon & sentuhan branding
        # Tambahkan footer di bagian bawah
        gr.HTML("""
            <footer id="footer">
                Transfer Energi Semesta Digital © 2024 | 🇮🇩 Untuk Indonesia Jaya!
            </footer>
            """)

    return demo




