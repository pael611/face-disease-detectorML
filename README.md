# Skin Condition Classification (MobileNetV2) + Skincare Recommendation

Repository ini berisi pipeline end-to-end untuk:

1) training model klasifikasi kondisi kulit wajah (6 kelas) menggunakan Transfer Learning MobileNetV2 (TensorFlow/Keras),
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
