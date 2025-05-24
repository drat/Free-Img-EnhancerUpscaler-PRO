![Free-Img-EnhanceUpscaler-PRO-a-Hugging-Face-Space-by-Deddy-05-24-2025_07_21_AM](https://github.com/user-attachments/assets/364df880-d3fd-422f-981d-11a9d7cb04e0)
```
 ▗▄▖ ▗▄▄▄▖    ▗▄▄▄▖▗▖  ▗▖ ▗▄▖  ▗▄▄▖▗▄▄▄▖    ▗▄▄▄▖▗▖  ▗▖▗▖ ▗▖ ▗▄▖ ▗▖  ▗▖ ▗▄▄▖▗▄▄▄▖▗▄▄▖ 
▐▌ ▐▌  █        █  ▐▛▚▞▜▌▐▌ ▐▌▐▌   ▐▌       ▐▌   ▐▛▚▖▐▌▐▌ ▐▌▐▌ ▐▌▐▛▚▖▐▌▐▌   ▐▌   ▐▌ ▐▌
▐▛▀▜▌  █        █  ▐▌  ▐▌▐▛▀▜▌▐▌▝▜▌▐▛▀▀▘    ▐▛▀▀▘▐▌ ▝▜▌▐▛▀▜▌▐▛▀▜▌▐▌ ▝▜▌▐▌   ▐▛▀▀▘▐▛▀▚▖
▐▌ ▐▌▗▄█▄▖    ▗▄█▄▖▐▌  ▐▌▐▌ ▐▌▝▚▄▞▘▐▙▄▄▖    ▐▙▄▄▖▐▌  ▐▌▐▌ ▐▌▐▌ ▐▌▐▌  ▐▌▝▚▄▄▖▐▙▄▄▖▐▌ ▐▌
 🚀 AI Image Toolkit
```

> **Aplikasi Gradio berbasis AI** untuk Peningkatan Gambar, Pemotongan, Segmentasi, dan Upscaling Cerdas.
---
![Free-Img-EnhanceUpscaler-PRO-a-Hugging-Face-Space-by-Deddy-05-24-2025_07_29_AM](https://github.com/user-attachments/assets/469fb9a9-336e-43ad-a218-6419cefa6217)
---
# Demo di HuggingFace
https://deddy-free-img-enhanceupscaler-pro.hf.space

## 📸 Tentang Proyek

**AI Image Toolkit** adalah aplikasi Gradio modular berbasis AI yang dirancang untuk tugas-tugas pasca-pemrosesan gambar. Baik Anda ingin meningkatkan kualitas foto buram, memotong objek secara cerdas, melakukan segmentasi semantik, atau memperbesar resolusi gambar untuk produksi — toolkit ini siap membantu.

Menggunakan model AI canggih (CNN & Transformer) di balik layar, alat ini menyajikan performa terbaik dengan antarmuka yang ramah pengguna dan backend yang modular.

---

## 🔍 Ringkasan Fitur

| Fitur        | Deskripsi |
|--------------|-----------|
| 🖼️ Enhance    | Meningkatkan kualitas gambar dengan AI (detail recovery, deblurring) |
| 🧠 Segment     | Segmentasi semantik menggunakan model pre-trained |
| ✂️ Crop        | Auto-cropping berdasarkan mask hasil segmentasi |
| 🆙 Upscale     | Meningkatkan resolusi gambar tanpa kehilangan detail |

---

## 🗂️ Struktur Berkas

```
📦 ai-image-toolkit/
├── app.py              # Peluncur utama Gradio
├── app_enhance.py      # Halaman Streamlit untuk enhancement
├── app_upscale.py      # Halaman Streamlit untuk upscaling
├── croper.py           # Logika cropping cerdas
├── enhance_utils.py    # Utilitas untuk pipeline enhancement
├── segment_utils.py    # Logika segmentasi semantik
├── requirements.txt    # Daftar dependensi
└── README.md
```

---

## ⚙️ Panduan Instalasi

### 🔧 Setup Lingkungan

```bash
git clone https://github.com/username/ai-image-toolkit.git
cd ai-image-toolkit
python -m venv venv
source venv/bin/activate   # atau venv\Scripts\activate di Windows
pip install -r requirements.txt
```

### ▶️ Jalankan Aplikasi

```bash
python app.py
```

---

## 🧪 Contoh Penggunaan

### 1. 🔬 Peningkatan Gambar

Unggah gambar buram atau resolusi rendah, lalu:
- Kontras ditingkatkan
- Detail dan tekstur dipertajam
- Hasil akhir berkualitas tinggi siap diunduh

### 2. 🧠 Segmentasi Semantik

Deteksi objek di gambar menggunakan model pre-trained (misalnya DeepLabV3), lalu:
- Tampilkan hasil mask segmentasi
- Ekspor sebagai file PNG

### 3. ✂️ Pemotongan Otomatis

Gunakan mask hasil segmentasi untuk:
- Mengekstrak objek dalam format PNG transparan
- Buat thumbnail produk secara otomatis

### 4. 📈 Upscaling Gambar

Perbesar gambar dari 256x256 jadi 2048x2048 dengan:
- Super-resolution berbasis AI
- Pemrosesan warna & tekstur canggih

---

## 🔍 Sorotan Teknologi

| Teknologi / Model  | Deskripsi |
|--------------------|-----------|
| 🔥 PyTorch          | Framework utama model AI |
| 🧠 DeepLabV3        | Segmentasi semantik |
| 📈 Real-ESRGAN      | Upscaling resolusi gambar |
| 🧰 OpenCV           | Manipulasi & preprocessing gambar |
| 🌐 Gradio        | Framework antarmuka pengguna |

---

## 📦 Dependensi

```
torch
torchvision==0.14.1
diffusers
transformers
accelerate
mediapipe
gradio
spaces
gfpgan
git+https://github.com/XPixelGroup/BasicSR@master
facexlib
realesrgan
```

*Instal otomatis dengan* `pip install -r requirements.txt`

---

## 🧠 Filosofi Desain

- 🔌 **Modularitas** — Mudah dikembangkan dengan model atau fitur baru
- 🧪 **Pipeline Reproducible** — Setiap proses dibungkus dalam fungsi utilitas
- ⚡ **Performa Cepat** — Gunakan GPU jika tersedia, fallback ke CPU
- 📐 **UI Responsif** — Siap untuk tampilan desktop dan layar sentuh

---

## 👨‍💻 Panduan Kontribusi

Ingin berkontribusi? Silakan!

1. Fork repositori ini
2. Buat branch fitur: `git checkout -b fitur/fitur-keren`
3. Commit perubahanmu: `git commit -m 'Tambah fitur keren'`
4. Push ke branch: `git push origin fitur/fitur-keren`
5. Buat Pull Request 🚀

---

## 🧾 Lisensi

Proyek ini dilisensikan dengan **MIT License** — lihat file [LICENSE](LICENSE) untuk detail lengkap.

---

## 🌍 Roadmap (Ide Pengembangan)

- ✅ Antarmuka drag & drop untuk batch processing
- 🚧 Tambahkan modul face enhancement (GFPGAN atau lainnya)
- 🚀 Ekspor langsung ke cloud atau Google Drive
- 📲 Deploy ke mobile via Streamlit Share / HuggingFace Spaces

---

## 🔗 Tautan Terkait

- [Dokumentasi Streamlit](https://docs.streamlit.io/)
- [PyTorch](https://pytorch.org/)
- [OpenCV](https://opencv.org/)
- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)
- [Paper DeepLabV3](https://arxiv.org/abs/1802.02611)

---

## 💬 Kontak

Dibuat dengan ❤️ oleh [Deddy](https://github.com/drat

> _"Biarkan model AI yang bekerja, antarmuka yang memandu."_ ✨
