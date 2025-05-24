"""
Image Upscaler App
------------------
Aplikasi AI berbasis Gradio yang memanfaatkan Stable Diffusion Upscaler untuk meningkatkan resolusi gambar.
Tersedia juga fitur segmentasi & restorasi area tertentu pada gambar (misal: wajah).
Aplikasi mendukung input prompt teks untuk conditioning hasil upscaling.

Created by _drat | 2025
"""

# Import library eksternal yang diperlukan
import requests
from PIL import Image
from io import BytesIO
from diffusers import StableDiffusionUpscalePipeline # Pipeline Stable Diffusion untuk upscaling gambar
import torch
import gradio as gr
import time
import spaces

# Import fungsi segmentasi dan restorasi (definisi di segment_utils.py)
from segment_utils import(
    segment_image,   # Untuk segmentasi area penting pada gambar (misal: wajah)
    restore_result,  # Untuk menggabungkan hasil upscaling dengan gambar asli
)

# Setup device: gunakan CUDA (GPU) jika tersedia, jika tidak fallback ke CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f'{device} is available')  # Debug: print device yang digunakan

# Load model Stable Diffusion Upscaler dari HuggingFace
model_id = "stabilityai/stable-diffusion-x4-upscaler"
upscale_pipe = StableDiffusionUpscalePipeline.from_pretrained(model_id, torch_dtype=torch.float16)
upscale_pipe = upscale_pipe.to(device)

# Default prompt dan kategori (untuk input Gradio)
DEFAULT_SRC_PROMPT = "a person with pefect face"
DEFAULT_CATEGORY = "face"

# Fungsi utama untuk membuat UI aplikasi Gradio
def create_demo() -> gr.Blocks:

    # --- [ Function Definitions Tetap Seperti Asli Anda ] ---
    @spaces.GPU(duration=30)
    def upscale_image(
        input_image: Image,
        prompt: str,
        num_inference_steps: int = 10,
    ):
        time_cost_str = ''
        run_task_time = 0
        run_task_time, time_cost_str = get_time_cost(run_task_time, time_cost_str)
        upscaled_image = upscale_pipe(
            prompt=prompt, 
            image=input_image,
            num_inference_steps=num_inference_steps,
        ).images[0]
        run_task_time, time_cost_str = get_time_cost(run_task_time, time_cost_str)
        return upscaled_image, time_cost_str

    def get_time_cost(run_task_time, time_cost_str):
        now_time = int(time.time()*1000)
        if run_task_time == 0:
            time_cost_str = 'start'
        else:
            if time_cost_str != '': 
                time_cost_str += f'-->'
            time_cost_str += f'{now_time - run_task_time}'
        run_task_time = now_time
        return run_task_time, time_cost_str

    # --- [ UI Section ] ---
    with gr.Blocks(css="creative_enhance.css") as demo:
        gr.HTML("""
        <div style='display:flex; justify-content:center;'>
            <img src='https://i.ibb.co/pvWSfsMJ/feature-image-upscaler.jpg' alt='Feature Image' style='height:200px; width:auto; border-radius:24px; box-shadow:0 4px 18px #ffa07430; border:2.5px solid #ffe0b3;' id='feature-image'/>
            </div>
        <div style='text-align:center;margin-top:6px;'>
            <span style='font-size:2.0rem;font-weight:800;color:#d84040;letter-spacing:0.04em;font-family:Quicksand,sans-serif;'>
                🔬 <b>AI Image Upscaler</b> <span style='font-size:1.2em;vertical-align:middle;'>🚀</span>
            </span>
            <br>
            <span style='font-size:1.1rem;color:#ff914d;font-family:Inter,sans-serif;'>Perbesar gambar <b>HD</b> otomatis, detail makin nyata!</span>
        </div>
        """)

        # Input prompt & parameter di satu card
        with gr.Row():
            with gr.Group(elem_id="control-card"):
                with gr.Row():
                    input_image_prompt = gr.Textbox(
                        lines=1, label="🎯 Prompt AI (opsional)",
                        value=DEFAULT_SRC_PROMPT, elem_id="input-image-prompt"
                    )
                    num_inference_steps = gr.Number(
                        label="⚙️ Steps (Quality)", value=5, elem_id="num-inference"
                    )
                    generate_size = gr.Number(
                        label="📐 Size (px)", value=512, elem_id="generate-size"
                    )
                g_btn = gr.Button("🪄 Upscale Sekarang", elem_id="upscale-btn")

        # Input & Output Gambar
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Group(elem_id="input-card"):
                    gr.Markdown("<div class='card-title'><span style='font-size:1.2em;'>🖼️</span> Gambar Asli</div>")
                    input_image = gr.Image(label="", type="pil", elem_id="input-image", show_label=False)
                    gr.Markdown("<div class='hint'><span style='font-size:1.1em;'>⬆️</span> JPG/PNG max 5MB</div>")
            with gr.Column(scale=1):
                with gr.Group(elem_id="output-card"):
                    gr.Markdown("<div class='card-title'><span style='font-size:1.2em;'>💡</span> Upscale Preview</div>")
                    restored_image = gr.Image(label="Hasil Akhir", format="png", type="pil", interactive=False)
                    origin_area_image = gr.Image(label="", format="png", type="pil", interactive=False, visible=False)
                    upscaled_image = gr.Image(label="Upscaled", format="png", type="pil", interactive=False)
                    download_path = gr.File(label="⬇️ Download Image", interactive=False, elem_id="download-btn")
                    gr.Markdown("<div class='hint'><span style='font-size:1.1em;'>💾</span> Download hasil upscale PNG</div>")
                    generated_cost = gr.Textbox(label="⏱️ Time (ms)", visible=True, interactive=False)

        # Hidden inputs untuk workflow
        category = gr.Textbox(label="Category", value=DEFAULT_CATEGORY, visible=False)
        mask_expansion = gr.Number(label="Mask Expansion", value=20, visible=False)
        mask_dilation = gr.Slider(minimum=0, maximum=10, value=2, step=1, label="Mask Dilation", visible=False)
        croper = gr.State()

        # Workflow chaining
        g_btn.click(
            fn=segment_image,
            inputs=[input_image, category, generate_size, mask_expansion, mask_dilation],
            outputs=[origin_area_image, croper],
        ).success(
            fn=upscale_image,
            inputs=[origin_area_image, input_image_prompt, num_inference_steps],
            outputs=[upscaled_image, generated_cost],
        ).success(
            fn=restore_result,
            inputs=[croper, category, upscaled_image],
            outputs=[restored_image, download_path],
        )

        gr.Markdown("""
        <div style='text-align:center;color:#aaa;font-size:0.98rem;margin-top:14px;'>
            &copy; 2025 <b>AI Image Upscaler</b> • Powered by <b>_drat</b> 🚀
        </div>
        """)

    return demo

