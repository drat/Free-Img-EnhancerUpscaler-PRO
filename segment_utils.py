"""
Segment Utils
-------------
Utilitas segmentasi gambar berbasis MediaPipe dan mask biner untuk aplikasi image enhancer/upscaler:
- Segmentasi area: wajah, rambut, pakaian, dll.
- Membuat dan mengelola mask hasil segmentasi serta integrasi dengan class Croper
- Mendukung proses restore hasil generate ke gambar asli

Created by _drat | 2025
"""

import numpy as np
import mediapipe as mp
import uuid
import os

from PIL import Image
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from scipy.ndimage import binary_dilation
from croper import Croper

# Inisialisasi model segmentasi MediaPipe (load tflite model untuk multiclass selfie segmentation)
segment_model = "checkpoints/selfie_multiclass_256x256.tflite"
base_options = python.BaseOptions(model_asset_path=segment_model)
options = vision.ImageSegmenterOptions(base_options=base_options, output_category_mask=True)
segmenter = vision.ImageSegmenter.create_from_options(options)

# Fungsi untuk mengembalikan hasil generate ke gambar asli, dengan mask transparan hasil segmentasi
def restore_result(croper, category, generated_image):
    square_length = croper.square_length
    generated_image = generated_image.resize((square_length, square_length))

    # Crop area hasil generate, lalu buat mask hasil restore
    cropped_generated_image = generated_image.crop((croper.square_start_x, croper.square_start_y, croper.square_end_x, croper.square_end_y))
    cropped_square_mask_image = get_restore_mask_image(croper, category, cropped_generated_image)

    # Tempel hasil generate ke gambar asli dengan mask transparan
    restored_image = croper.input_image.copy()
    restored_image.paste(cropped_generated_image, (croper.origin_start_x, croper.origin_start_y), cropped_square_mask_image)

    extension = 'png'
    tmpPrefix = "/tmp/gradio/"
    # targetDir = f"{tmpPrefix}output/"
    targetDir = "output/"
    if not os.path.exists(targetDir):
        os.makedirs(targetDir)

    path = f"{targetDir}{uuid.uuid4()}.{extension}"
    restored_image.save(path, quality=100)

    return restored_image, path

# Fungsi utama segmentasi gambar sesuai kategori
def segment_image(input_image, category, input_size, mask_expansion, mask_dilation):
    mask_size = int(input_size)
    mask_expansion = int(mask_expansion)

    # Konversi input image ke format yang sesuai untuk MediaPipe
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.asarray(input_image))
    segmentation_result = segmenter.segment(image)
    category_mask = segmentation_result.category_mask
    category_mask_np = category_mask.numpy_view()

    # Pilih fungsi pembentukan mask sesuai kategori segmentasi
    if category == "hair":
        target_mask = get_hair_mask(category_mask_np, mask_dilation)
    elif category == "clothes":
        target_mask = get_clothes_mask(category_mask_np, mask_dilation)
    elif category == "face":
        target_mask = get_face_mask(category_mask_np, mask_dilation)
    else:
        target_mask = get_face_mask(category_mask_np, mask_dilation)
    
    # Buat objek Croper dengan mask hasil segmentasi, untuk crop & resize
    croper = Croper(input_image, target_mask, mask_size, mask_expansion)
    croper.corp_mask_image()
    origin_area_image = croper.resized_square_image

    return origin_area_image, croper  # Kembalikan area hasil crop + objek croper

# Fungsi pembuat mask khusus area wajah (+opsi dilasi mask)
def get_face_mask(category_mask_np, dilation=1):
    face_skin_mask = category_mask_np == 3
    if dilation > 0:
        face_skin_mask = binary_dilation(face_skin_mask, iterations=dilation)
    return face_skin_mask

# Fungsi pembuat mask khusus area baju (dan kulit tubuh)
def get_clothes_mask(category_mask_np, dilation=1):
    body_skin_mask = category_mask_np == 2
    clothes_mask = category_mask_np == 4
    combined_mask = np.logical_or(body_skin_mask, clothes_mask)
    combined_mask = binary_dilation(combined_mask, iterations=4)
    if dilation > 0:
        combined_mask = binary_dilation(combined_mask, iterations=dilation)
    return combined_mask

# Fungsi pembuat mask khusus area rambut
def get_hair_mask(category_mask_np, dilation=1):
    hair_mask = category_mask_np == 1
    if dilation > 0:
        hair_mask = binary_dilation(hair_mask, iterations=dilation)
    return hair_mask

# Fungsi untuk membuat mask kombinasi hasil generate dan area awal (untuk proses restore)
def get_restore_mask_image(croper, category, generated_image):
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.asarray(generated_image))
    segmentation_result = segmenter.segment(image)
    category_mask = segmentation_result.category_mask
    category_mask_np = category_mask.numpy_view()

    if category == "hair":
        target_mask = get_hair_mask(category_mask_np, 0)
    elif category == "clothes":
        target_mask = get_clothes_mask(category_mask_np, 0)
    elif category == "face":
        target_mask = get_face_mask(category_mask_np, 0)
    
    # Gabungkan mask hasil segmentasi dan mask area awal croper
    combined_mask = np.logical_or(target_mask, croper.corp_mask)
    mask_image = Image.fromarray((combined_mask * 255).astype(np.uint8))
    return mask_image
