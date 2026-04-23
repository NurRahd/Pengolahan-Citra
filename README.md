# Histogram Lab — Pengolahan Citra Digital RGB

Aplikasi web interaktif untuk analisis dan pemrosesan histogram gambar RGB. Dibangun dengan Python Flask, NumPy, dan PIL untuk demonstrasi konsep pengolahan citra digital.

## 🎯 Fitur

- **Analisis Histogram RGB**: Unggah gambar dan lihat histogram terpisah untuk channel Merah (R), Hijau (G), dan Biru (B)
- **Histogram Equalization**: Tingkatkan kontras gambar dengan equalisasi histogram per channel
- **Histogram Matching/Specification**: Sesuaikan distribusi warna gambar sumber agar mendekati gambar referensi
- **Visualisasi Interaktif**: Chart histogram dengan Chart.js, tabel perhitungan detail, dan preview gambar real-time
- **Interface Web Modern**: UI responsif dengan tema gelap dan animasi smooth

## 🛠️ Teknologi

- **Backend**: Python Flask
- **Pemrosesan Gambar**: NumPy, Pillow (PIL)
- **Frontend**: HTML5, CSS3, JavaScript ES6
- **Visualisasi**: Chart.js 4.4.0
- **Font**: Inter & JetBrains Mono dari Google Fonts

## 📋 Persyaratan Sistem

- Python 3.7+
- Browser modern dengan dukungan ES6

## 🚀 Instalasi & Menjalankan

1. **Clone atau download** file `app.py` dan `requirements.txt`

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Jalankan aplikasi**:
   ```bash
   python app.py
   ```

4. **Buka browser** dan akses `http://localhost:5000`

## 📖 Cara Penggunaan

1. **Langkah 1**: Unggah gambar RGB untuk dianalisis histogramnya
2. **Langkah 2**: Jalankan equalization untuk meningkatkan kontras
3. **Langkah 3**: Unggah gambar referensi
4. **Langkah 4**: Lakukan histogram matching untuk menyesuaikan warna

Setiap langkah menyediakan tabel perhitungan detail dan visualisasi histogram sebelum/sesudah pemrosesan.

## 📊 Algoritma

- **Histogram Calculation**: Hitung frekuensi piksel per intensitas (0-255) per channel
- **Equalization**: Gunakan Cumulative Distribution Function (CDF) dengan formula:
  ```
  s = round( (CDF(r) - CDF_min) / (N - CDF_min) * 255 )
  ```
- **Matching**: Cari intensitas referensi dengan CDF terdekat untuk setiap nilai sumber

## 📁 Struktur Proyek

```
files/
├── app.py          # Aplikasi Flask utama
├── requirements.txt # Dependencies Python
└── README.md       # Dokumentasi ini
```

## 🤝 Kontribusi

Proyek ini dibuat untuk keperluan akademik (PPM - Pengolahan Citra Digital). Kode dapat dimodifikasi dan dikembangkan lebih lanjut.

## 📄 Lisensi

MIT License - bebas digunakan untuk keperluan edukasi dan non-komersial.
