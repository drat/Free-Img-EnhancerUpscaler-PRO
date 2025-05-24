"""
Enhance Utils
-------------
Utility functions untuk meningkatkan kualitas gambar dengan menggunakan model AI:
- Real-ESRGAN untuk upscaling gambar umum
- GFPGAN untuk enhancement wajah (face restoration)

Created by _drat | 2025
"""

import os
import torch
import cv2
import numpy as np
import subprocess

from PIL import Image
from gfpgan.utils import GFPGANer
from basicsr.archs.srvgg_arch import SRVGGNetCompact
from realesrgan.utils import RealESRGANer

# Fungsi untuk menjalankan perintah shell (misal: pip, wget)
def runcmd(cmd, verbose = False, *args, **kwargs):
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

# Debug: Menampilkan daftar library yang terinstall di environment
runcmd("pip freeze")

# Download model GFPGAN jika belum ada di direktori saat ini
if not os.path.exists('GFPGANv1.4.pth'):
    runcmd("wget https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth -P .")

# Download model RealESRGAN jika belum ada di direktori saat ini
if not os.path.exists('realesr-general-x4v3.pth'):
    runcmd("wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth -P .")

# Inisialisasi model Real-ESRGAN (untuk upscaling gambar umum)
model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
model_path = 'realesr-general-x4v3.pth'
half = True if torch.cuda.is_available() else False  # Pakai half precision jika tersedia GPU
upsampler = RealESRGANer(
    scale=4,
    model_path=model_path,
    model=model,
    tile=0,
    tile_pad=10,
    pre_pad=0,
    half=half
)

# Inisialisasi enhancer wajah (GFPGAN)
face_enhancer = GFPGANer(
    model_path='GFPGANv1.4.pth',
    upscale=1,                 # Default tidak melakukan upscaling (hanya enhance wajah)
    arch='clean',
    channel_multiplier=2
)

# Fungsi utama enhancement gambar
def enhance_image(
    pil_image: Image,     # Input gambar dalam format PIL
    enhance_face: bool = False, # Apakah perlu enhance wajah?
):
    # Konversi PIL image ke array BGR (OpenCV)
    img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    h, w = img.shape[0:2]
    # Optional: jika gambar kecil, perbesar dulu agar hasil enhancement lebih maksimal
    if h < 300:
        img = cv2.resize(img, (w * 2, h * 2), interpolation=cv2.INTER_LANCZOS4)
    
    # Enhance menggunakan GFPGAN (wajah) atau Real-ESRGAN (umum)
    if enhance_face:
        # Hanya enhance bagian wajah utama di tengah gambar
        _, _, output = face_enhancer.enhance(
            img, has_aligned=False, only_center_face=True, paste_back=True
        )
    else:
        # Upscale seluruh gambar dengan Real-ESRGAN (outscale=2x)
        output, _ = upsampler.enhance(img, outscale=2)
    
    # Konversi hasil output ke format PIL RGB untuk siap digunakan di aplikasi
    pil_output = Image.fromarray(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))

    return pil_output
