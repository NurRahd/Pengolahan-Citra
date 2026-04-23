# Histogram Lab — Image Processing Web App

## Fitur
- Upload gambar & konversi ke **grayscale**
- Tampilkan **histogram** citra
- **Histogram Equalization** menggunakan CDF
- Upload gambar referensi
- **Histogram Specification (Matching)**
- Tampilkan hasil sebelum & sesudah (gambar + histogram)

## Cara Menjalankan

### 1. Install dependensi
```bash
pip install -r requirements.txt
```

### 2. Jalankan server
```bash
python app.py
```

### 3. Buka browser
Kunjungi: http://localhost:5000

## Alur Penggunaan
1. Upload gambar sumber → klik **Konversi ke Grayscale**
2. Klik **Histogram Equalization** untuk menyamakan distribusi histogram
3. Upload gambar referensi → klik **Histogram Matching** untuk mencocokkan histogram sumber ke referensi
4. Lihat perbandingan gambar dan histogram di panel kanan

## Implementasi Algoritma
- **Grayscale**: `PIL.Image.convert('L')` (luminosity formula)
- **Histogram Equalization**: CDF normalisasi → LUT mapping
- **Histogram Matching**: Mapping CDF sumber ke CDF referensi menggunakan nearest-neighbor
