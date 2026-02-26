# Skin Condition Classification — MobileNetV2

> Klasifikasi kondisi kulit wajah berbasis Deep Learning dengan sistem rekomendasi produk skincare.

Proyek ini menyajikan pipeline Machine Learning end-to-end untuk klasifikasi kondisi kulit wajah berbasis citra dengan arsitektur MobileNetV2 (transfer learning). Dataset disusun dari kombinasi data publik (Kaggle) dan data kurasi manual (Google Images serta koleksi gambar lokal/path), kemudian diproses melalui tahapan validasi data, preprocessing terstandar, augmentasi, two-stage training, evaluasi kuantitatif, dan ekspor model ke format ONNX.

Dokumen ini ditulis dengan orientasi teknis-ilmiah agar dapat digunakan sebagai referensi implementasi, reproduksibilitas eksperimen, dan baseline pengembangan lanjutan.

Referensi implementasi utama: `prediction_model.py`.

---

## Daftar Isi

1. [Gambaran Proyek](#1-gambaran-proyek)
2. [Struktur Direktori](#2-struktur-direktori)
3. [Dataset](#3-dataset)
4. [Pipeline Preprocessing](#4-pipeline-preprocessing)
5. [Augmentasi Data](#5-augmentasi-data)
6. [Strategi Mengatasi Class Imbalance](#6-strategi-mengatasi-class-imbalance)
7. [Arsitektur Model](#7-arsitektur-model)
8. [Strategi Training (Two-Stage)](#8-strategi-training-two-stage)
9. [Evaluasi Model](#9-evaluasi-model)
10. [Penyimpanan dan Ekspor Model](#10-penyimpanan-dan-ekspor-model)
11. [Inferensi dan Rekomendasi Produk](#11-inferensi-dan-rekomendasi-produk)
12. [Cara Menjalankan Proyek](#12-cara-menjalankan-proyek)
13. [Dependensi Utama](#13-dependensi-utama)
14. [Catatan dan Batasan](#14-catatan-dan-batasan)

---

## 1. Gambaran Proyek

### 1.1 Tujuan

Membangun model image classification yang mampu:

1. Memprediksi kondisi kulit wajah dari sebuah gambar ke dalam 6 kelas.
2. Menampilkan confidence score hasil prediksi.
3. Memberikan rekomendasi produk skincare berdasarkan label prediksi.

### 1.2 Kelas Kondisi Kulit

| Indeks | Label |
|---|---|
| 0 | Acne |
| 1 | Blackheads |
| 2 | Dark Spots |
| 3 | Normal Skin |
| 4 | Oily Skin |
| 5 | Wrinkles |

### 1.3 Ringkasan Alur End-to-End

1. Kumpulkan file gambar dari folder dataset per kelas.
2. Filter file valid berdasarkan ekstensi gambar.
3. Lakukan stratified split train/validation (80/20).
4. Bangun pipeline `tf.data` + preprocessing PIL + augmentasi.
5. Latih model dalam 2 tahap (224 lalu 320).
6. Evaluasi performa pada validation set.
7. Simpan model Keras dan ekspor ke ONNX.
8. Gunakan model untuk prediksi gambar baru dan tampilkan rekomendasi produk.

---

## 2. Struktur Direktori

```text
PMLDI/
├── prediction_model.py
├── README.md
├── requirments.txt
├── classification_report.txt
├── confusion_matrix.png
├── best_skin_model_stage1.h5
├── best_skin_model.h5
├── final_skin_model.h5
├── best_skin_model_320.onnx
├── dataset/
│   ├── Acne/
│   ├── Blackheads/
│   ├── Dark Spots/
│   ├── Normal Skin/
│   ├── Oily Skin/
│   └── Wrinkles/
├── skincare_product/
│   ├── treatment.csv
│   ├── treatment.json
│   └── gambar_produk/
└── debug_face_crop/
```

Keterangan penting:

- `prediction_model.py`: script utama training, evaluasi, inferensi, dan ekspor ONNX.
- `dataset/`: data gambar terstruktur folder-per-class.
- `skincare_product/treatment.csv`: mapping label kulit ke daftar produk rekomendasi.
- `debug_face_crop/`: output debug preview face-crop (jika mode debug aktif).

---

## 3. Dataset

### 3.1 Format Dataset

Dataset menggunakan struktur folder-per-class (satu folder = satu label).

Ekstensi file yang diproses:

- `.jpg`
- `.jpeg`
- `.png`
- `.bmp`
- `.gif`

File di luar ekstensi tersebut akan diabaikan.

### 3.2 Sumber Data

Dataset pada proyek ini berasal dari dua sumber utama:

1. **Dataset publik dari Kaggle**
      - Digunakan sebagai sumber data primer untuk memperkaya variasi kondisi kulit.
      - Kontribusi utama: konsistensi label awal dan jumlah sampel yang lebih besar.

2. **Kurasi manual**
      - **Google Images**: pengumpulan gambar tambahan untuk meningkatkan variasi domain visual.
      - **Gambar lokal/path**: koleksi gambar pribadi/arsip lokal yang relevan dengan kelas target.
      - Kontribusi utama: memperluas keragaman distribusi data terhadap kondisi nyata.

### 3.3 Prosedur Kurasi Manual (Google + Path)

Untuk menjaga kualitas data dan mengurangi noise label, prosedur kurasi dilakukan sebagai berikut:

- Seleksi gambar yang menampilkan area wajah/kulit secara jelas.
- Eliminasi gambar blur berat, resolusi terlalu rendah, atau artefak kompresi ekstrem.
- Penyelarasan label kelas agar konsisten dengan 6 kelas target.
- Penataan file ke struktur folder-per-class sebelum proses split.

### 3.4 Potensi Bias Data

Karena data berasal dari kombinasi Kaggle + kurasi manual, terdapat potensi:

- bias pencahayaan (lighting conditions),
- bias perangkat kamera,
- bias demografi/tipe kulit,
- duplikasi semantik antar sumber.

Langkah mitigasi yang diterapkan pada pipeline ini adalah augmentasi ringan, stratified split, dan oversampling berbasis kelas.

### 3.5 Split Train/Validation

Split dilakukan menggunakan:

```python
train_test_split(filepaths, labels, test_size=0.2, random_state=42, stratify=labels)
```

- `test_size=0.2` → 80% train, 20% validation
- `stratify=labels` → distribusi kelas terjaga di train dan validation

---

## 4. Pipeline Preprocessing

Pipeline preprocessing pada script menggunakan pendekatan robust berbasis PIL:

1. Baca gambar dengan `PIL.Image.open`.
2. Koreksi orientasi EXIF dengan `ImageOps.exif_transpose`.
3. Konversi ke RGB.
4. (Opsional) auto face-crop.
5. Resize dengan `ImageOps.pad` (menjaga aspek rasio + padding).
6. Konversi ke `float32`.
7. Normalisasi input MobileNetV2 menggunakan `preprocess_input`.

### 4.1 Auto Face-Crop (Opsional)

Konfigurasi utama:

```python
ENABLE_AUTO_FACE_CROP = True
FACE_CROP_MARGIN = 0.25
```

Detail:

- Deteksi wajah menggunakan OpenCV Haar Cascade.
- Jika terdeteksi lebih dari satu wajah, diambil wajah dengan area terbesar.
- Bounding box diperluas dengan margin agar area wajah tidak terlalu ketat.
- Jika deteksi gagal, gambar asli tetap dipakai (fallback aman).

### 4.2 Debug Face-Crop

Konfigurasi debug:

```python
DEBUG_FACE_CROP_PREVIEW = False
DEBUG_FACE_CROP_DIR = 'debug_face_crop'
DEBUG_FACE_CROP_MAX_SAVES = 40
```

Jika diaktifkan, sistem menyimpan panel perbandingan:

- sebelum crop
- setelah crop
- hasil final setelah resize

---

## 5. Augmentasi Data

Augmentasi diterapkan hanya pada data training di pipeline `tf.data`:

- `RandomFlip('horizontal')`
- `RandomRotation(0.03)`
- `RandomTranslation(0.03, 0.03)`
- `RandomZoom((-0.05, 0.05), (-0.05, 0.05))`

Validation set tidak diaugmentasi agar evaluasi tetap objektif.

---

## 6. Strategi Mengatasi Class Imbalance

Proyek ini menggunakan dua pendekatan:

### 6.1 Class Weights (Analisis)

Class weights dihitung dengan:

```python
compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
```

### 6.2 Oversampling di `tf.data` (Dipakai saat Training)

- Dataset dibagi per kelas.
- Tiap dataset kelas di-`repeat()`.
- Digabung dengan `tf.data.experimental.sample_from_datasets` menggunakan bobot setara antar kelas.

Dampak:

- Distribusi kelas saat training lebih seimbang.
- Mengurangi bias model ke kelas mayoritas.

---

## 7. Arsitektur Model

Arsitektur model:

1. Backbone: `MobileNetV2(include_top=False, weights='imagenet')`
2. `GlobalAveragePooling2D`
3. `Dense(128, activation='relu')`
4. `Dropout(0.3)`
5. Output: `Dense(6, activation='softmax')`

Catatan implementasi:

- Input model bersifat dinamis `(None, None, 3)`.
- Backbone awalnya dibekukan untuk stage warmup.

---

## 8. Strategi Training (Two-Stage)

### 8.1 Stage 1 — Warmup

Tujuan: melatih classifier head terlebih dahulu agar stabil.

- Resolusi: `224x224`
- Batch size: `16`
- Backbone: frozen (`trainable=False`)
- Learning rate: `1e-4`
- Epoch: `6`
- Checkpoint: `best_skin_model_stage1.h5`

### 8.2 Stage 2 — Fine-tuning

Tujuan: meningkatkan detail representasi dengan resolusi lebih tinggi.

- Resolusi: `320x320`
- Batch size: `8`
- Backbone: dibuka (`trainable=True`)
- Layer BatchNorm tetap frozen untuk stabilitas
- Learning rate: `1e-5`
- Epoch: `20`
- Checkpoint terbaik: `best_skin_model.h5`

Rasional ilmiah two-stage training:

- **Stage 1 (frozen backbone)** menstabilkan pembelajaran classifier head sebelum propagasi gradien ke seluruh backbone.
- **Stage 2 (fine-tuning + LR kecil)** memfasilitasi adaptasi fitur tingkat tinggi tanpa merusak representasi umum dari pretraining ImageNet.
- **BatchNorm tetap frozen** untuk mengurangi instabilitas statistik batch saat ukuran batch kecil.

### 8.3 Callback Training

- `EarlyStopping` (monitor `val_loss`, restore best weights)
- `ReduceLROnPlateau`
- `ModelCheckpoint`

---

## 9. Evaluasi Model

Evaluasi dilakukan pada validation set menggunakan:

- Confusion Matrix (`confusion_matrix.png`)
- Classification Report (`classification_report.txt`)

### 9.1 Hasil Classification Report (Saat Ini)

```text
              precision    recall  f1-score   support

        Acne     0.8571    0.9231    0.8889        39
  Blackheads     0.9048    0.9500    0.9268        40
  Dark Spots     0.8649    0.7805    0.8205        41
 Normal Skin     0.7500    0.7778    0.7636        27
   Oily Skin     0.7692    0.6667    0.7143        30
    Wrinkles     0.9643    1.0000    0.9818        54

    accuracy                         0.8701       231
   macro avg     0.8517    0.8497    0.8493       231
weighted avg     0.8679    0.8701    0.8677       231
```

Interpretasi ilmiah singkat:

- **Akurasi keseluruhan**: 87.01% pada 231 sampel validasi.
- **Macro F1-score**: 0.8493 menunjukkan performa rata-rata antarkelas relatif seimbang, namun masih ada gap di kelas minor/ambigu.
- **Weighted F1-score**: 0.8677 menunjukkan performa global baik dengan mempertimbangkan distribusi support tiap kelas.
- **Kelas paling kuat**: Wrinkles (recall 1.0000, F1 0.9818).
- **Kelas menantang**: Oily Skin (recall 0.6667) dan Normal Skin (F1 0.7636), mengindikasikan overlap fitur visual antar kelas.

---

## 10. Penyimpanan dan Ekspor Model

### 10.1 Artefak Model

- `best_skin_model_stage1.h5` (checkpoint terbaik stage 1)
- `best_skin_model.h5` (checkpoint terbaik stage 2)
- `final_skin_model.h5` (snapshot akhir training)

### 10.2 Ekspor ONNX

Ekspor menggunakan `tf2onnx` dari model terbaik stage 2:

- Output: `best_skin_model_320.onnx`
- Input signature: `(None, 320, 320, 3)`
- Opset: `15`

---

## 11. Inferensi dan Rekomendasi Produk

### 11.1 Prediksi Gambar Baru

Fungsi utama:

- `load_and_preprocess_image(img_path, target_size=None)`
- `predict_skin_condition(model, img_path, categories)`

Output prediksi:

- indeks kelas
- nama kelas
- confidence (%)

### 11.2 Rekomendasi Produk

Fungsi:

- `show_recommendations(predicted_label)`

Sumber data rekomendasi:

- `skincare_product/treatment.csv`

Sistem akan menampilkan:

- brand
- nama produk
- harga
- link
- gambar produk (jika tersedia)

---

## 12. Cara Menjalankan Proyek

### 12.1 Instal Dependensi

Gunakan file yang ada di repo:

```bash
pip install -r requirments.txt
```

> Catatan: nama file di repo adalah `requirments.txt` (bukan `requirements.txt`).

### 12.2 Jalankan Pipeline Utama

```bash
python prediction_model.py
```

Script akan menjalankan:

1. Persiapan data dan split
2. Stage 1 training
3. Stage 2 training
4. Evaluasi
5. Simpan model
6. Ekspor ONNX
7. Contoh prediksi dan rekomendasi produk

### 12.3 Output yang Dihasilkan

- `best_skin_model_stage1.h5`
- `best_skin_model.h5`
- `final_skin_model.h5`
- `classification_report.txt`
- `confusion_matrix.png`
- `best_skin_model_320.onnx`

---

## 13. Dependensi Utama

Dependensi inti yang digunakan oleh proyek:

- TensorFlow / Keras
- NumPy
- Pandas
- Pillow
- OpenCV
- scikit-learn
- Matplotlib
- Seaborn
- tf2onnx
- onnx

Versi lengkap paket tersedia di `requirments.txt`.

---

## 14. Catatan dan Batasan

1. Sumber data merupakan gabungan **Kaggle** dan **kurasi manual (Google Images + gambar lokal/path)**, sehingga heterogenitas domain visual cukup tinggi.
2. Kualitas label pada data manual sangat bergantung pada proses anotasi/validasi manusia.
3. Model ini adalah sistem pendukung keputusan berbasis citra, **bukan** alat diagnosis medis.
4. Rekomendasi produk bersifat rule-based berdasarkan label klasifikasi, belum memodelkan profil pengguna secara personal.
5. Generalisasi lintas domain (kamera, pencahayaan, demografi) tetap perlu diuji lebih lanjut melalui external validation set.

Pengembangan lanjutan yang direkomendasikan:

- audit kualitas label dan analisis inter-annotator agreement,
- evaluasi fairness lintas subpopulasi,
- kalibrasi probabilitas (temperature scaling),
- deployment benchmark ONNX Runtime untuk latency dan throughput.

