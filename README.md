# 🧴 Skin Condition Classification Using Deep Learning with Transfer Learning MobileNetV2

## Sistem Klasifikasi Kondisi Kulit Wajah dan Rekomendasi Produk Skincare Berbasis Deep Learning

---

## 📋 Abstrak

Proyek ini mengembangkan sistem klasifikasi kondisi kulit wajah berbasis **Deep Learning** menggunakan arsitektur **Transfer Learning MobileNetV2**. Sistem mampu mengklasifikasikan 6 jenis kondisi kulit wajah: **Acne (Jerawat)**, **Blackheads (Komedo)**, **Dark Spots (Flek Hitam)**, **Normal Skin (Kulit Normal)**, **Oily Skin (Kulit Berminyak)**, dan **Wrinkles (Kerutan)**. Selain klasifikasi, sistem juga menyediakan **rekomendasi produk skincare** yang sesuai berdasarkan hasil prediksi. Model dilatih menggunakan dataset citra kulit wajah dengan teknik **data augmentation** dan **oversampling** untuk menangani ketidakseimbangan kelas. Model disimpan dalam format Keras (`.h5`) dan diekspor ke format **ONNX** untuk keperluan deployment lintas platform.

**Kata Kunci:** *Deep Learning, Transfer Learning, MobileNetV2, Klasifikasi Kulit, Computer Vision, Skincare Recommendation, TensorFlow, ONNX*

---

## 📚 Daftar Isi

1. [Pendahuluan](#-pendahuluan)
2. [Latar Belakang](#-latar-belakang)
3. [Tujuan Penelitian](#-tujuan-penelitian)
4. [Arsitektur Sistem](#-arsitektur-sistem)
5. [Metodologi](#-metodologi)
6. [Dataset](#-dataset)
7. [Preprocessing Data](#-preprocessing-data)
8. [Arsitektur Model](#-arsitektur-model)
9. [Training Pipeline](#-training-pipeline)
10. [Hasil dan Evaluasi](#-hasil-dan-evaluasi)
11. [Fitur Rekomendasi Produk](#-fitur-rekomendasi-produk)
12. [Struktur Repository](#-struktur-repository)
13. [Requirements & Instalasi](#-requirements--instalasi)
14. [Cara Penggunaan](#-cara-penggunaan)
15. [Ekspor Model ke ONNX](#-ekspor-model-ke-onnx)
16. [Kesimpulan](#-kesimpulan)
17. [Pengembangan Selanjutnya](#-pengembangan-selanjutnya)
18. [Referensi](#-referensi)

---

## 📖 Pendahuluan

Kesehatan kulit wajah merupakan aspek penting dalam perawatan diri yang mempengaruhi kepercayaan diri seseorang. Berbagai kondisi kulit seperti jerawat, komedo, flek hitam, kulit berminyak, dan kerutan memerlukan penanganan yang berbeda-beda. Identifikasi kondisi kulit yang tepat menjadi langkah awal dalam pemilihan produk skincare yang sesuai.

Proyek ini mengembangkan sistem berbasis **Artificial Intelligence (AI)** yang dapat mengklasifikasikan kondisi kulit wajah secara otomatis melalui analisis citra menggunakan teknik **Deep Learning**. Dengan memanfaatkan arsitektur **MobileNetV2** yang telah dilatih pada dataset ImageNet (*pre-trained model*), sistem ini menerapkan konsep **Transfer Learning** untuk mencapai akurasi tinggi meskipun dengan dataset yang terbatas.

---

## 🎯 Latar Belakang

### Permasalahan
1. **Kesulitan Identifikasi Kondisi Kulit**: Banyak orang kesulitan mengenali jenis dan kondisi kulit mereka sendiri
2. **Pemilihan Produk yang Tidak Tepat**: Tanpa pengetahuan yang cukup, pemilihan produk skincare sering tidak sesuai dengan kebutuhan kulit
3. **Keterbatasan Akses ke Dermatolog**: Tidak semua orang memiliki akses mudah ke konsultasi dermatolog profesional
4. **Class Imbalance pada Dataset**: Dataset kondisi kulit sering tidak seimbang antar kelas

### Solusi yang Ditawarkan
- Sistem klasifikasi otomatis berbasis citra dengan Deep Learning
- Rekomendasi produk skincare berdasarkan hasil klasifikasi
- Model yang ringan dan dapat di-deploy di berbagai platform (menggunakan ONNX)
- Teknik oversampling untuk mengatasi ketidakseimbangan data

---

## 🎯 Tujuan Penelitian

1. **Tujuan Utama:**
   - Mengembangkan model Deep Learning untuk klasifikasi kondisi kulit wajah dengan akurasi tinggi
   
2. **Tujuan Khusus:**
   - Menerapkan Transfer Learning menggunakan arsitektur MobileNetV2
   - Mengatasi masalah class imbalance dengan teknik oversampling
   - Mengintegrasikan sistem rekomendasi produk skincare
   - Mengekspor model ke format ONNX untuk deployment lintas platform

---

## 🏗️ Arsitektur Sistem

```
┌─────────────────────────────────────────────────────────────────────┐
│                     SISTEM KLASIFIKASI KULIT                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ┌──────────┐    ┌──────────────┐    ┌─────────────────────┐      │
│   │  Input   │    │ Preprocessing │    │   MobileNetV2       │      │
│   │  Image   │───▶│  - Resize    │───▶│   (Feature          │      │
│   │ (224x224)│    │  - Normalize │    │    Extraction)      │      │
│   └──────────┘    └──────────────┘    └──────────┬──────────┘      │
│                                                   │                 │
│                                                   ▼                 │
│   ┌──────────────────────────────────────────────────────────┐     │
│   │                  Classification Head                      │     │
│   │  GlobalAveragePooling2D → Dense(128) → Dropout → Softmax │     │
│   └──────────────────────────────┬───────────────────────────┘     │
│                                  │                                  │
│                                  ▼                                  │
│   ┌──────────────────────────────────────────────────────────┐     │
│   │                    Output Prediction                      │     │
│   │  [Acne | Blackheads | Dark Spots | Normal | Oily | Wrinkles]   │
│   └──────────────────────────────┬───────────────────────────┘     │
│                                  │                                  │
│                                  ▼                                  │
│   ┌──────────────────────────────────────────────────────────┐     │
│   │              Product Recommendation System                │     │
│   │         (treatment.csv / treatment.json)                  │     │
│   └──────────────────────────────────────────────────────────┘     │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 📊 Metodologi

### Alur Penelitian

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Dataset   │────▶│ Preprocessing│────▶│  Training   │────▶│  Evaluation │
│  Collection │     │  & Augment  │     │   Model     │     │   & Test    │
└─────────────┘     └─────────────┘     └─────────────┘     └──────┬──────┘
                                                                    │
┌─────────────┐     ┌─────────────┐     ┌─────────────┐            │
│   Deploy    │◀────│   Export    │◀────│    Save     │◀───────────┘
│   (ONNX)    │     │   to ONNX   │     │   Model     │
└─────────────┘     └─────────────┘     └─────────────┘
```

### Tahapan Metodologi:

1. **Pengumpulan Data**: Dataset citra kulit wajah dengan 6 kategori
2. **Preprocessing**: Resize, normalisasi, dan augmentasi data
3. **Stratified Split**: Pembagian data train/validation (80:20) dengan stratifikasi
4. **Oversampling**: Teknik balancing untuk mengatasi class imbalance
5. **Transfer Learning**: Fine-tuning MobileNetV2 pre-trained
6. **Training**: Pelatihan model dengan callbacks optimal
7. **Evaluasi**: Confusion matrix dan classification report
8. **Export**: Konversi model ke format ONNX

---

## 📁 Dataset

### Struktur Dataset

```
dataset/
├── Acne/           # Citra kulit berjerawat
├── Blackheads/     # Citra kulit dengan komedo
├── Dark Spots/     # Citra kulit dengan flek hitam
├── Normal Skin/    # Citra kulit normal/sehat
├── Oily Skin/      # Citra kulit berminyak
└── Wrinkles/       # Citra kulit dengan kerutan
```

### Deskripsi Kelas

| No | Kelas | Deskripsi | Karakteristik |
|----|-------|-----------|---------------|
| 1 | **Acne** | Jerawat | Peradangan pada kulit, pustula, papula |
| 2 | **Blackheads** | Komedo | Pori-pori tersumbat, titik hitam |
| 3 | **Dark Spots** | Flek Hitam | Hiperpigmentasi, noda gelap |
| 4 | **Normal Skin** | Kulit Normal | Tekstur halus, tidak berminyak berlebih |
| 5 | **Oily Skin** | Kulit Berminyak | Kilap berlebih, pori-pori besar |
| 6 | **Wrinkles** | Kerutan | Garis halus, lipatan kulit |

### Distribusi Data (Validation Set)

| Kelas | Jumlah Sample (Support) |
|-------|------------------------|
| Acne | 25 |
| Blackheads | 41 |
| Dark Spots | 71 |
| Normal Skin | 80 |
| Oily Skin | 79 |
| Wrinkles | 78 |
| **Total** | **374** |

### Format Citra yang Didukung
- JPEG (`.jpg`, `.jpeg`)
- PNG (`.png`)
- BMP (`.bmp`)
- GIF (`.gif`)

---

## ⚙️ Preprocessing Data

### 1. Image Resizing
Semua citra di-resize ke ukuran **224 × 224 piksel** sesuai dengan input yang dibutuhkan MobileNetV2.

```python
img_size = 224
target_size = (img_size, img_size)
```

### 2. Normalisasi
Menggunakan preprocessing function dari MobileNetV2 yang melakukan normalisasi piksel ke rentang [-1, 1].

```python
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
```

### 3. Data Augmentation
Teknik augmentasi untuk meningkatkan variasi data training:

```python
datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2,
    rotation_range=20,        # Rotasi ±20 derajat
    width_shift_range=0.2,    # Pergeseran horizontal 20%
    height_shift_range=0.2,   # Pergeseran vertikal 20%
    horizontal_flip=True,     # Flip horizontal
    brightness_range=[0.8, 1.2],  # Variasi brightness
    shear_range=0.2,          # Shear transformation
    zoom_range=0.2            # Zoom in/out 20%
)
```

### 4. Robust Image Decoding
Menggunakan **Pillow (PIL)** untuk decoding citra yang lebih robust dibandingkan TensorFlow native decoder:

```python
def pil_decode_and_resize(path):
    path_str = path.numpy().decode("utf-8")
    with Image.open(path_str) as img:
        img = img.convert("RGB")
        img = img.resize((img_size, img_size))
        arr = np.asarray(img, dtype=np.float32)
        return arr
```

### 5. Stratified Split
Pembagian data dengan stratifikasi untuk memastikan distribusi kelas yang proporsional:

```python
X_train, X_val, y_train, y_val = train_test_split(
    filepaths, labels, 
    test_size=0.2, 
    random_state=42, 
    stratify=labels
)
```

### 6. Oversampling untuk Class Balancing
Teknik oversampling menggunakan `tf.data.experimental.sample_from_datasets`:

```python
# Dataset per-kelas dengan repeat() untuk oversampling
per_class_datasets = []
for cls_idx, cls_name in enumerate(categories):
    cls_files = [fp for fp, lab in zip(X_train, y_train) if lab == cls_idx]
    ds = tf.data.Dataset.from_tensor_slices((cls_files, [cls_idx] * len(cls_files)))
    ds = ds.shuffle(max(8*BATCH_SIZE, len(cls_files)))
    ds = ds.repeat()  # Infinite repeat untuk sampling
    ds = ds.map(_load_and_preprocess, num_parallel_calls=AUTOTUNE)
    per_class_datasets.append(ds)

# Sampling dengan bobot seimbang
weights = [1.0/num_classes] * num_classes
balanced_ds = tf.data.experimental.sample_from_datasets(per_class_datasets, weights=weights)
```

---

## 🧠 Arsitektur Model

### Base Model: MobileNetV2

**MobileNetV2** dipilih karena:
- **Efisiensi**: Arsitektur ringan dengan performa tinggi
- **Pre-trained**: Sudah dilatih pada ImageNet (1.4 juta citra, 1000 kelas)
- **Transfer Learning**: Cocok untuk fine-tuning pada dataset kecil
- **Mobile-friendly**: Dapat di-deploy pada perangkat mobile

```python
base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,      # Tanpa fully connected layer asli
    weights='imagenet'       # Pre-trained weights
)
base_model.trainable = False  # Freeze base model
```

### Custom Classification Head

```python
inputs = Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)
x = GlobalAveragePooling2D()(x)     # Reduce spatial dimensions
x = Dense(128, activation='relu')(x) # Hidden layer
x = Dropout(0.3)(x)                  # Regularization
outputs = Dense(6, activation='softmax')(x)  # Output 6 kelas
model = Model(inputs, outputs)
```

### Ringkasan Arsitektur

```
Model: "functional"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         [(None, 224, 224, 3)]     0         
_________________________________________________________________
mobilenetv2_1.00_224         (None, 7, 7, 1280)        2,257,984 
_________________________________________________________________
global_average_pooling2d     (None, 1280)              0         
_________________________________________________________________
dense (Dense)                (None, 128)               163,968   
_________________________________________________________________
dropout (Dropout)            (None, 128)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 6)                 774       
=================================================================
Total params: 2,422,726
Trainable params: 164,742
Non-trainable params: 2,257,984
_________________________________________________________________
```

### Hyperparameters

| Parameter | Nilai |
|-----------|-------|
| Input Size | 224 × 224 × 3 |
| Batch Size | 64 |
| Learning Rate | 1e-4 (0.0001) |
| Optimizer | Adam |
| Loss Function | Sparse Categorical Crossentropy |
| Hidden Units | 128 |
| Dropout Rate | 0.3 (30%) |
| Epochs | 20 (max) |

---

## 🏋️ Training Pipeline

### Kompilasi Model

```python
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

### Callbacks

1. **EarlyStopping**: Menghentikan training jika tidak ada improvement
   ```python
   EarlyStopping(
       monitor='val_loss', 
       patience=10, 
       restore_best_weights=True
   )
   ```

2. **ReduceLROnPlateau**: Menurunkan learning rate saat plateau
   ```python
   ReduceLROnPlateau(
       monitor='val_loss', 
       factor=0.2, 
       patience=5, 
       min_lr=1e-6
   )
   ```

3. **ModelCheckpoint**: Menyimpan model terbaik
   ```python
   ModelCheckpoint(
       'best_skin_model.h5',
       monitor='val_accuracy',
       save_best_only=True,
       mode='max',
       verbose=1
   )
   ```

### Proses Training

```python
steps_per_epoch = max(1, sum(class_counts) // BATCH_SIZE)

history = model.fit(
    train_ds_balanced,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_ds,
    epochs=20,
    callbacks=[early_stopping, checkpoint, reduce_lr]
)
```

---

## 📈 Hasil dan Evaluasi

### Classification Report

```
              precision    recall  f1-score   support

        Acne     0.8519    0.9200    0.8846        25
  Blackheads     0.7955    0.8537    0.8235        41
  Dark Spots     0.8235    0.7887    0.8058        71
 Normal Skin     0.7910    0.6625    0.7211        80
   Oily Skin     0.6824    0.7342    0.7073        79
    Wrinkles     0.8916    0.9487    0.9193        78

    accuracy                         0.7995       374
   macro avg     0.8060    0.8180    0.8103       374
weighted avg     0.7998    0.7995    0.7977       374
```

### Interpretasi Metrik

| Metrik | Deskripsi |
|--------|-----------|
| **Precision** | Ketepatan prediksi positif (TP / (TP + FP)) |
| **Recall** | Kemampuan mendeteksi kelas sebenarnya (TP / (TP + FN)) |
| **F1-Score** | Harmonic mean dari precision dan recall |
| **Support** | Jumlah sampel per kelas |
| **Accuracy** | Ketepatan keseluruhan |

### Analisis Per-Kelas

| Kelas | Precision | Recall | F1-Score | Analisis |
|-------|-----------|--------|----------|----------|
| **Acne** | 85.19% | 92.00% | 88.46% | ✅ Performa sangat baik |
| **Blackheads** | 79.55% | 85.37% | 82.35% | ✅ Performa baik |
| **Dark Spots** | 82.35% | 78.87% | 80.58% | ✅ Performa baik |
| **Normal Skin** | 79.10% | 66.25% | 72.11% | ⚠️ Recall perlu ditingkatkan |
| **Oily Skin** | 68.24% | 73.42% | 70.73% | ⚠️ Precision perlu ditingkatkan |
| **Wrinkles** | 89.16% | 94.87% | 91.93% | ✅ Performa terbaik |

### Akurasi Keseluruhan

| Metrik | Nilai |
|--------|-------|
| **Accuracy** | **79.95%** |
| **Macro Avg F1** | 81.03% |
| **Weighted Avg F1** | 79.77% |

### Visualisasi Training

Model menghasilkan kurva training (accuracy dan loss) yang menunjukkan konvergensi yang baik tanpa overfitting berlebih berkat penerapan:
- Dropout regularization
- Early stopping
- Learning rate reduction

---

## 🛍️ Fitur Rekomendasi Produk

### Database Produk

Sistem dilengkapi database produk skincare dalam format CSV dan JSON (`skincare_product/treatment.csv` dan `treatment.json`) yang berisi **67 produk** dari berbagai brand ternama.

### Struktur Data Produk

```json
{
  "Id": 1,
  "Label": "Treatment/Acne",
  "Brand": "COSRX",
  "Product Name": "Acne Pimple Master Patch",
  "Price": "Rp59,000",
  "Tags": "Acne",
  "Links": "https://www.sociolla.com/..."
}
```

### Brand yang Tersedia

- COSRX, AVOSKIN, SOMETHINC, WHITELAB
- MEDIHEAL, LACOCO, DEAR KLAIRS, SKINTIFIC
- BIODERMA, CETAPHIL, LA ROCHE POSAY, NEUTROGENA
- AZARINE, NIVEA, EMINA, dan lainnya

### Kategori Produk per Kondisi Kulit

| Kondisi Kulit | Jenis Produk Rekomendasi |
|---------------|--------------------------|
| **Acne** | Acne Patch, Toner, Essence, Serum, Face Mask |
| **Blackheads** | Exfoliating Toner, AHA/BHA Serum, Clay Mask |
| **Dark Spots** | Brightening Toner, Vitamin C Serum, Whitening Essence |
| **Normal Skin** | Cleanser, Moisturizer, Sunscreen |
| **Oily Skin** | Oil-Free Cleanser, Mattifying Moisturizer, Lightweight Sunscreen |
| **Wrinkles** | Retinol Toner, Anti-aging Serum, Vitamin C Serum |

### Fungsi Rekomendasi

```python
def show_recommendations(predicted_label):
    df = pd.read_csv('skincare_product/treatment.csv')
    produk = df[df['Tags'].str.lower() == predicted_label.lower()]
    
    print(f"\nRekomendasi produk untuk '{predicted_label}':")
    for _, row in produk.iterrows():
        print(f"- {row['Brand']} | {row['Product Name']} | {row['Price']}")
        print(f"  Link: {row['Links']}")
```

---

## 📂 Struktur Repository

```
PMLDI/
│
├── 📊 Model Files
│   ├── best_skin_model.h5              # Model Keras (checkpoint terbaik)
│   └── best_skin_model.onnx            # Model ONNX untuk deployment
│
├── 📈 Evaluation Results
│   ├── classification_report.txt       # Laporan metrik evaluasi
│   └── confusion_matrix.png            # Visualisasi confusion matrix
│
├── 📓 Code Files
│   ├── prediction_model.py             # Script utama training & inference
│   ├── prediction_model.ipynb          # Versi Jupyter Notebook
│   └── prediction_model_converted.ipynb # Notebook hasil konversi
│
├── 📁 Dataset
│   └── dataset/
│       ├── Acne/
│       ├── Blackheads/
│       ├── Dark Spots/
│       ├── Normal Skin/
│       ├── Oily Skin/
│       └── Wrinkles/
│
├── 🛍️ Product Recommendations
│   └── skincare_product/
│       ├── treatment.csv               # Database produk (CSV)
│       ├── treatment.json              # Database produk (JSON)
│       └── gambar_produk/              # Gambar produk
│
├── 📝 Documentation
│   ├── README.md                       # Dokumentasi proyek
│   └── requirments.txt                 # Dependencies Python
│
└── 🖼️ Test Images (optional)
    └── *.jpg                           # Gambar untuk testing
```

---

## 📦 Requirements & Instalasi

### System Requirements
- Python 3.8+ (Recommended: Python 3.10)
- GPU dengan CUDA support (optional, untuk training lebih cepat)
- Minimum 8GB RAM

### Instalasi Dependencies

```cmd
pip install -r requirments.txt
```

### Library Utama

| Library | Versi | Fungsi |
|---------|-------|--------|
| TensorFlow | 2.x | Deep Learning framework |
| Keras | (included) | High-level neural network API |
| NumPy | 1.x | Numerical computing |
| Pandas | 1.x/2.x | Data manipulation |
| Matplotlib | 3.x | Visualisasi |
| Seaborn | 0.x | Statistical visualization |
| Scikit-learn | 1.x | Machine learning utilities |
| Pillow (PIL) | 9.x/10.x | Image processing |
| tf2onnx | 1.x | Konversi TensorFlow ke ONNX |
| ONNX | 1.x | Open Neural Network Exchange |

---

## 🚀 Cara Penggunaan

### 1. Training Model

**Via Python Script:**
```cmd
python prediction_model.py
```

**Via Jupyter Notebook:**
1. Buka `prediction_model.ipynb`
2. Jalankan semua cell secara berurutan

### 2. Prediksi Single Image

```python
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np

# Load model
model = tf.keras.models.load_model('best_skin_model.h5')
categories = ['Acne', 'Blackheads', 'Dark Spots', 'Normal Skin', 'Oily Skin', 'Wrinkles']

# Load dan preprocess gambar
img_path = 'path/to/test_image.jpg'
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = preprocess_input(img_array)
img_array = tf.expand_dims(img_array, axis=0)

# Prediksi
predictions = model.predict(img_array)
predicted_class = tf.argmax(predictions, axis=1).numpy()[0]
confidence = float(np.max(predictions)) * 100

print(f"Kondisi Kulit: {categories[predicted_class]}")
print(f"Confidence: {confidence:.2f}%")
```

### 3. Prediksi dengan Rekomendasi Produk

```python
from prediction_model import predict_skin_condition, show_recommendations, categories

model = tf.keras.models.load_model('best_skin_model.h5')
img_path = 'test_image.jpg'

# Prediksi
pred_idx, pred_name, conf = predict_skin_condition(model, img_path, categories)
print(f"Prediksi: {pred_name} ({conf:.2f}%)")

# Tampilkan rekomendasi produk
show_recommendations(pred_name)
```

---

## 🔄 Ekspor Model ke ONNX

Model diekspor ke format **ONNX (Open Neural Network Exchange)** untuk deployment lintas platform (web, mobile, embedded systems).

```python
import tf2onnx
import onnx
import tensorflow as tf

# Load model Keras
model = tf.keras.models.load_model('best_skin_model.h5')

# Definisikan input signature
spec = (tf.TensorSpec((None, 224, 224, 3), tf.float32, name="input_1"),)

# Konversi ke ONNX
output_path = 'best_skin_model.onnx'
model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, output_path=output_path)
onnx.save_model(model_proto, output_path)

print(f"Model berhasil diekspor ke {output_path}")
```

### Penggunaan Model ONNX

```python
import onnxruntime as ort
import numpy as np

# Load ONNX model
session = ort.InferenceSession('best_skin_model.onnx')

# Prepare input
input_name = session.get_inputs()[0].name
input_data = np.random.randn(1, 224, 224, 3).astype(np.float32)

# Run inference
result = session.run(None, {input_name: input_data})
```

---

## 📝 Kesimpulan

### Pencapaian Proyek

1. **Model Deep Learning** berhasil dikembangkan untuk klasifikasi 6 kondisi kulit wajah
2. **Akurasi 79.95%** dicapai menggunakan Transfer Learning MobileNetV2
3. **Teknik Oversampling** berhasil mengatasi masalah class imbalance
4. **Sistem Rekomendasi** terintegrasi dengan database 67 produk skincare
5. **Model ONNX** tersedia untuk deployment lintas platform

### Kelebihan Sistem

- ✅ Menggunakan Transfer Learning untuk efisiensi training
- ✅ Robust image decoding dengan Pillow
- ✅ Balanced training dengan oversampling
- ✅ Multiple callbacks untuk optimasi training
- ✅ Cross-platform deployment dengan ONNX
- ✅ Integrated product recommendation

### Keterbatasan

- ⚠️ Akurasi pada kelas Normal Skin dan Oily Skin masih perlu ditingkatkan
- ⚠️ Dataset terbatas pada 6 kategori kondisi kulit
- ⚠️ Belum mendukung deteksi multiple conditions pada satu gambar

---

## 🔮 Pengembangan Selanjutnya

1. **Fine-tuning Base Model**: Unfreeze beberapa layer teratas MobileNetV2
2. **Augmentasi Tambahan**: Eksperimen dengan teknik augmentasi lanjutan
3. **Ensemble Methods**: Kombinasi beberapa model untuk meningkatkan akurasi
4. **Web/Mobile App**: Deployment sebagai aplikasi dengan antarmuka pengguna
5. **Real-time Detection**: Implementasi deteksi kondisi kulit secara real-time
6. **Expanded Dataset**: Penambahan lebih banyak data untuk kelas dengan performa rendah
7. **Multi-label Classification**: Mendukung deteksi multiple conditions

---

## 📚 Referensi

1. Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L. C. (2018). **MobileNetV2: Inverted residuals and linear bottlenecks**. *Proceedings of the IEEE conference on computer vision and pattern recognition*, 4510-4520.

2. Howard, A. G., Zhu, M., Chen, B., Kalenichenko, D., Wang, W., Weyand, T., ... & Adam, H. (2017). **MobileNets: Efficient convolutional neural networks for mobile vision applications**. *arXiv preprint arXiv:1704.04861*.

3. Russakovsky, O., Deng, J., Su, H., Krause, J., Satheesh, S., Ma, S., ... & Fei-Fei, L. (2015). **ImageNet large scale visual recognition challenge**. *International journal of computer vision*, 115(3), 211-252.

4. TensorFlow Documentation: https://www.tensorflow.org/
5. ONNX Documentation: https://onnx.ai/

---

## 👨‍💻 Author

**Project:** Skin Condition Classification Using Deep Learning  
**Purpose:** Educational and Research (Dicoding Submission)  
**License:** Educational and Non-Commercial Use

---

## 📄 Lisensi

Proyek ini dikembangkan untuk keperluan **edukasi dan penelitian**. Penggunaan untuk tujuan komersial memerlukan izin terlebih dahulu.

---

<div align="center">

**⭐ Jika proyek ini bermanfaat, berikan bintang pada repository ini! ⭐**

</div>
