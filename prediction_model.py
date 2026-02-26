# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: main-ds
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Skin Condition Classification dengan MobileNetV2
# Pipeline ini mencakup: preprocessing data, two-stage training, evaluasi, dan ekspor ONNX.

# %% 1. Import Library
import os
import glob
import hashlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image

from PIL import Image, ImageOps
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# %% 2. Konfigurasi Global
# ---------------------------------------------------------------
# Konfigurasi utama: path dataset, ukuran gambar, dan batch size.
# Two-stage training digunakan agar backbone terlatih secara bertahap:
#   Stage 1 - warmup classifier head pada resolusi kecil (224) agar cepat konvergen.
#   Stage 2 - fine-tune seluruh model pada resolusi lebih besar (320) agar lebih detail.
# ---------------------------------------------------------------

dataset_path = 'dataset'
categories   = ['Acne', 'Blackheads', 'Dark Spots', 'Normal Skin', 'Oily Skin', 'Wrinkles']
num_classes  = len(categories)

IMG_SIZE_STAGE1   = 224
BATCH_SIZE_STAGE1 = 16

IMG_SIZE_STAGE2   = 320
BATCH_SIZE_STAGE2 = 8

# Ukuran & batch yang dipakai saat inferensi / ekspor (selaras dengan Stage 2)
img_size   = IMG_SIZE_STAGE2
BATCH_SIZE = BATCH_SIZE_STAGE2

# %% 3. Konfigurasi Face-Crop (Opsional)
# ---------------------------------------------------------------
# Jika aktif, setiap gambar akan di-crop ke area wajah terbesar
# menggunakan Haar Cascade dari OpenCV sebelum diproses lebih lanjut.
# Ini membantu mengurangi noise dari latar belakang.
# ---------------------------------------------------------------

ENABLE_AUTO_FACE_CROP  = True
FACE_CROP_MARGIN       = 0.25   # perbesar kotak wajah 25% ke setiap sisi

# Debug: simpan preview hasil face-crop ke disk (set False untuk produksi)
DEBUG_FACE_CROP_PREVIEW   = False
DEBUG_FACE_CROP_DIR       = 'debug_face_crop'
DEBUG_FACE_CROP_MAX_SAVES = 40

_FACE_CASCADE          = None   # cache objek CascadeClassifier
_DEBUG_FACE_CROP_SAVED = 0      # counter gambar debug yang sudah disimpan


# %% 4. Fungsi Debug Preview Face-Crop

def _maybe_save_face_crop_preview(
    *,
    source_path: str,
    before_img,
    cropped_img,
    final_img,
    target_img_size: int,
    resample,
):
    """Simpan preview tiga-panel (asli | crop | final) jika mode debug aktif.

    Preview disimpan di direktori DEBUG_FACE_CROP_DIR dan dibatasi
    hingga DEBUG_FACE_CROP_MAX_SAVES file agar tidak membebani I/O.
    Fungsi ini tidak boleh melempar exception ke pipeline utama.
    """
    global _DEBUG_FACE_CROP_SAVED

    if not DEBUG_FACE_CROP_PREVIEW:
        return

    try:
        if _DEBUG_FACE_CROP_SAVED >= DEBUG_FACE_CROP_MAX_SAVES:
            return

        idx = _DEBUG_FACE_CROP_SAVED
        _DEBUG_FACE_CROP_SAVED += 1

        os.makedirs(DEBUG_FACE_CROP_DIR, exist_ok=True)

        stem     = os.path.basename(source_path)
        h        = hashlib.md5(source_path.encode('utf-8', errors='ignore')).hexdigest()[:8]
        out_path = os.path.join(DEBUG_FACE_CROP_DIR, f"{idx:04d}_s{target_img_size}_{stem}_{h}.jpg")

        # Normalisasi tiga gambar ke ukuran yang sama agar perbandingan jelas
        a = ImageOps.pad(before_img,  (target_img_size, target_img_size), method=resample)
        b = ImageOps.pad(cropped_img, (target_img_size, target_img_size), method=resample)
        c = final_img

        grid = Image.new('RGB', (target_img_size * 3, target_img_size))
        grid.paste(a, (0, 0))
        grid.paste(b, (target_img_size, 0))
        grid.paste(c, (target_img_size * 2, 0))
        grid.save(out_path, quality=92)

    except Exception:
        # Error di fungsi debug tidak boleh menghentikan proses training
        return


# %% 5. Fungsi Face-Crop Otomatis

def _get_face_cascade():
    """Muat dan cache CascadeClassifier OpenCV (lazy loading).

    Mengembalikan None jika OpenCV tidak terinstall atau file cascade tidak ditemukan,
    sehingga pipeline tetap berjalan tanpa face-crop.
    """
    global _FACE_CASCADE
    if _FACE_CASCADE is not None:
        return _FACE_CASCADE
    try:
        import cv2
        cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
        cascade = cv2.CascadeClassifier(cascade_path)
        if cascade is None or cascade.empty():
            _FACE_CASCADE = None
            return None
        _FACE_CASCADE = cascade
        return _FACE_CASCADE
    except Exception:
        _FACE_CASCADE = None
        return None


def _auto_face_crop_pil(img, margin: float = FACE_CROP_MARGIN):
    """Crop gambar PIL ke area wajah terbesar yang terdeteksi.

    Jika OpenCV tidak tersedia atau tidak ada wajah yang terdeteksi,
    fungsi ini mengembalikan gambar asli tanpa perubahan.

    Args:
        img:    Objek PIL.Image dalam format RGB.
        margin: Rasio padding di luar kotak wajah (default 0.25 = 25%).

    Returns:
        PIL.Image: Gambar hasil crop, atau gambar asli jika deteksi gagal.
    """
    import cv2

    cascade = _get_face_cascade()
    if cascade is None:
        return img

    try:
        rgb = np.asarray(img)
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            return img

        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape[:2]
        min_side = max(30, int(min(h, w) * 0.15))

        faces = cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(min_side, min_side),
        )
        if faces is None or len(faces) == 0:
            return img

        # Pilih wajah dengan area terbesar
        x, y, fw, fh = max(faces, key=lambda b: b[2] * b[3])
        mx = int(fw * margin)
        my = int(fh * margin)
        x1, y1 = max(0, x - mx), max(0, y - my)
        x2, y2 = min(w, x + fw + mx), min(h, y + fh + my)

        if x2 <= x1 or y2 <= y1:
            return img

        return img.crop((x1, y1, x2, y2))

    except Exception:
        return img


# %% 6. Persiapan Dataset — Kumpulkan File & Label

# Ekstensi gambar yang didukung oleh decoder TensorFlow
image_exts   = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.gif")
valid_suffix = (".jpg", ".jpeg", ".png", ".bmp", ".gif")

filepaths, labels = [], []
for cls_idx, cls_name in enumerate(categories):
    cls_dir   = os.path.join(dataset_path, cls_name)
    cls_files = []
    for ext in image_exts:
        cls_files.extend(glob.glob(os.path.join(cls_dir, ext)))
    filepaths += cls_files
    labels    += [cls_idx] * len(cls_files)

# Filter defensif: buang file dengan ekstensi tidak dikenal
filtered  = [(fp, lab) for fp, lab in zip(filepaths, labels) if fp.lower().endswith(valid_suffix)]
filepaths = [fp  for fp, _   in filtered]
labels    = [lab for _,  lab in filtered]


# %% 7. Split Data — Train / Validasi (Stratified 80/20)

X_train, X_val, y_train, y_val = train_test_split(
    filepaths, labels,
    test_size=0.2,
    random_state=42,
    stratify=labels,   # jaga distribusi kelas tetap seimbang di kedua split
)

# Tampilkan distribusi kelas sebelum oversampling
print("Distribusi data training (sebelum oversampling):")
unique, counts = np.unique(y_train, return_counts=True)
for u, c in zip(unique, counts):
    print(f"  {categories[u]}: {c}")


# %% 8. Class Weights (referensi; oversampling aktif di pipeline tf.data)
# Dihitung untuk referensi, meskipun pipeline utama menggunakan oversampling.

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train,
)
class_weights = dict(enumerate(class_weights))
print("Class weights:", class_weights)


# %% 9. Augmentasi & Pipeline tf.data

AUTOTUNE = tf.data.AUTOTUNE


def _make_data_augmentation(name: str = 'data_augmentation'):
    """Buat layer augmentasi Keras Sequential (flip, rotasi, translasi, zoom)."""
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip('horizontal'),
            tf.keras.layers.RandomRotation(0.03, fill_mode='reflect'),
            tf.keras.layers.RandomTranslation(0.03, 0.03, fill_mode='reflect'),
            tf.keras.layers.RandomZoom((-0.05, 0.05), (-0.05, 0.05), fill_mode='reflect'),
        ],
        name=name,
    )


def _make_preprocess_fns(target_img_size: int, apply_augment: bool):
    """Buat sepasang fungsi preprocessing untuk training dan validasi.

    Fungsi training menerapkan augmentasi; fungsi validasi tidak.
    Keduanya menggunakan PIL untuk decoding yang lebih presisi
    (mendukung EXIF transpose dan face-crop otomatis).

    Args:
        target_img_size: Ukuran sisi gambar output (px).
        apply_augment:   Aktifkan augmentasi pada subset training.

    Returns:
        Tuple[Callable, Callable]: (fn_train, fn_val)
    """
    data_augmentation = _make_data_augmentation(f'data_augmentation_{target_img_size}')

    def pil_decode_and_resize(path):
        """Baca, crop wajah (opsional), dan resize gambar menggunakan PIL."""
        path_str = path.numpy().decode('utf-8')
        try:
            resample = Image.Resampling.BICUBIC
        except AttributeError:
            resample = Image.BICUBIC

        with Image.open(path_str) as img:
            img = ImageOps.exif_transpose(img)   # koreksi orientasi EXIF
            img = img.convert('RGB')

            before = img

            if ENABLE_AUTO_FACE_CROP:
                img = _auto_face_crop_pil(img, margin=FACE_CROP_MARGIN)

            cropped = img

            # Pad-resize agar aspek rasio tidak distorsi
            try:
                img = ImageOps.pad(img, (target_img_size, target_img_size), method=resample)
            except Exception:
                img = img.resize((target_img_size, target_img_size), resample=resample)

            _maybe_save_face_crop_preview(
                source_path=path_str,
                before_img=before,
                cropped_img=cropped,
                final_img=img,
                target_img_size=target_img_size,
                resample=resample,
            )

            return np.asarray(img, dtype=np.float32)

    def _load_and_preprocess(path, label, training: bool):
        img = tf.py_function(func=pil_decode_and_resize, inp=[path], Tout=tf.float32)
        img = tf.reshape(img, [target_img_size, target_img_size, 3])

        if training and apply_augment:
            # Normalisasi ke [0,1] → augmentasi → kembalikan ke skala asli
            img01 = tf.clip_by_value(img / 255.0, 0.0, 1.0)
            img01 = data_augmentation(img01, training=True)
            img   = tf.clip_by_value(img01 * 255.0, 0.0, 255.0)

        img = preprocess_input(img)   # normalisasi MobileNetV2 [-1, 1]
        return img, label

    def _load_and_preprocess_train(path, label):
        return _load_and_preprocess(path, label, training=True)

    def _load_and_preprocess_val(path, label):
        return _load_and_preprocess(path, label, training=False)

    return _load_and_preprocess_train, _load_and_preprocess_val


def build_balanced_datasets(target_img_size: int, batch_size: int):
    """Buat tf.data Dataset dengan oversampling seimbang antar kelas.

    Setiap kelas diambil dengan peluang yang sama menggunakan
    `sample_from_datasets`, terlepas dari jumlah sampel aslinya.

    Args:
        target_img_size: Resolusi gambar (piksel, persegi).
        batch_size:      Ukuran batch.

    Returns:
        Tuple: (train_ds, val_ds, class_counts, steps_per_epoch)
    """
    fn_train, fn_val = _make_preprocess_fns(
        target_img_size=target_img_size,
        apply_augment=True,
    )

    # Dataset validasi — tanpa shuffle dan augmentasi
    val_ds = (
        tf.data.Dataset.from_tensor_slices((X_val, y_val))
        .map(fn_val, num_parallel_calls=AUTOTUNE)
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )

    # Dataset per kelas untuk oversampling proporsional
    per_class_datasets = []
    class_counts       = []
    for cls_idx, cls_name in enumerate(categories):
        cls_files = [fp for fp, lab in zip(X_train, y_train) if lab == cls_idx]
        class_counts.append(len(cls_files))

        ds = (
            tf.data.Dataset.from_tensor_slices((cls_files, [cls_idx] * len(cls_files)))
            .shuffle(max(8 * batch_size, len(cls_files)))
            .repeat()   # ulangi terus agar oversampling bisa berjalan
            .map(fn_train, num_parallel_calls=AUTOTUNE)
        )
        per_class_datasets.append(ds)

    # Gabungkan semua kelas dengan bobot seragam
    weights     = [1.0 / num_classes] * num_classes
    balanced_ds = tf.data.experimental.sample_from_datasets(per_class_datasets, weights=weights)
    train_ds    = balanced_ds.batch(batch_size).prefetch(AUTOTUNE)

    steps_per_epoch = max(1, sum(class_counts) // batch_size)
    return train_ds, val_ds, class_counts, steps_per_epoch


print("Pipeline tf.data siap (decoder: PIL, mode: balanced oversampling).")

# %% [markdown]
# ## Membangun Model

# %% 10. Definisi Arsitektur — MobileNetV2 + Classifier Head

# MobileNetV2 sebagai backbone pre-trained (ImageNet), input dinamis agar
# bisa dipakai di dua stage resolusi berbeda tanpa rebuild model.
base_model = MobileNetV2(
    input_shape=(None, None, 3),
    include_top=False,
    weights='imagenet',
)
base_model.trainable = False   # bekukan backbone untuk Stage 1

# Classifier head kustom di atas backbone
inputs  = Input(shape=(None, None, 3))
x       = base_model(inputs, training=False)
x       = GlobalAveragePooling2D()(x)
x       = Dense(128, activation='relu')(x)
x       = Dropout(0.3)(x)
outputs = Dense(num_classes, activation='softmax')(x)
model   = Model(inputs, outputs)

model.summary()

# %% 11. Callbacks Training

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=5,
    min_lr=1e-6,
)

# Simpan bobot terbaik berdasarkan val_accuracy selama Stage 2
checkpoint = ModelCheckpoint(
    'best_skin_model.h5',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1,
)

# %% 12. Two-Stage Training

# --- Stage 1: Warmup Classifier Head (backbone beku, resolusi 224) ---
print(f"\nStage 1: warmup @ {IMG_SIZE_STAGE1}×{IMG_SIZE_STAGE1}, batch={BATCH_SIZE_STAGE1}")

train_ds_s1, val_ds_s1, counts_s1, steps_s1 = build_balanced_datasets(
    target_img_size=IMG_SIZE_STAGE1,
    batch_size=BATCH_SIZE_STAGE1,
)
print(f"  steps_per_epoch Stage 1: {steps_s1}")

checkpoint_stage1 = ModelCheckpoint(
    'best_skin_model_stage1.h5',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1,
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
)

history_stage1 = model.fit(
    train_ds_s1,
    steps_per_epoch=steps_s1,
    validation_data=val_ds_s1,
    epochs=6,
    callbacks=[
        EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
        checkpoint_stage1,
    ],
)

# --- Stage 2: Fine-Tune Seluruh Model (backbone dibuka, resolusi 320) ---
print(f"\nStage 2: fine-tune @ {IMG_SIZE_STAGE2}×{IMG_SIZE_STAGE2}, batch={BATCH_SIZE_STAGE2}")

train_ds_balanced, val_ds, class_counts, steps_per_epoch = build_balanced_datasets(
    target_img_size=IMG_SIZE_STAGE2,
    batch_size=BATCH_SIZE_STAGE2,
)
print(f"  steps_per_epoch Stage 2: {steps_per_epoch}")

# Buka semua layer backbone kecuali BatchNormalization
# (BN tetap beku agar statistik batch stabil saat fine-tuning)
base_model.trainable = True
for layer in base_model.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),   # LR lebih kecil untuk fine-tune
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
)

history_balanced = model.fit(
    train_ds_balanced,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_ds,
    epochs=20,
    callbacks=[early_stopping, checkpoint, reduce_lr],
)

# %% 13. Visualisasi Kurva Training

def plot_training_history(history):
    """Plot kurva akurasi dan loss training vs validasi."""
    acc      = history.history.get('accuracy', [])
    val_acc  = history.history.get('val_accuracy', [])
    loss     = history.history.get('loss', [])
    val_loss = history.history.get('val_loss', [])

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(acc,     label='Train Accuracy')
    plt.plot(val_acc, label='Val Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(loss,     label='Train Loss')
    plt.plot(val_loss, label='Val Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()


plot_training_history(history_balanced)

# %% 14. Fungsi Preprocessing & Prediksi Gambar Baru

def load_and_preprocess_image(img_path, target_size=None):
    """Baca dan proses satu gambar untuk inferensi.

    Menggunakan PIL sebagai decoder utama (mendukung face-crop otomatis).
    Jika PIL gagal, fallback ke Keras image loader.

    Args:
        img_path:    Path ke file gambar.
        target_size: Tuple (height, width). Default: (img_size, img_size).

    Returns:
        tf.Tensor: Batch gambar dengan shape (1, H, W, 3), ternormalisasi.
    """
    if target_size is None:
        target_size = (img_size, img_size)

    try:
        try:
            resample = Image.Resampling.BICUBIC
        except AttributeError:
            resample = Image.BICUBIC

        with Image.open(img_path) as img:
            img = ImageOps.exif_transpose(img)
            img = img.convert('RGB')

            before = img
            if ENABLE_AUTO_FACE_CROP:
                img = _auto_face_crop_pil(img, margin=FACE_CROP_MARGIN)
            cropped = img

            img = ImageOps.pad(img, target_size, method=resample)

            _maybe_save_face_crop_preview(
                source_path=str(img_path),
                before_img=before,
                cropped_img=cropped,
                final_img=img,
                target_img_size=int(target_size[0]),
                resample=resample,
            )
            img_array = np.asarray(img, dtype=np.float32)

    except Exception:
        # Fallback: gunakan Keras loader jika PIL gagal
        img       = image.load_img(img_path, target_size=target_size)
        img_array = image.img_to_array(img)

    img_array = preprocess_input(img_array)
    return tf.expand_dims(img_array, axis=0)


def predict_skin_condition(model, img_path, categories):
    """Prediksi kondisi kulit dari satu gambar.

    Args:
        model:      Model Keras yang sudah dilatih.
        img_path:   Path ke file gambar.
        categories: List nama kelas sesuai urutan output model.

    Returns:
        Tuple[int, str, float]: (indeks kelas, nama kelas, confidence %)
    """
    img_array       = load_and_preprocess_image(img_path)
    predictions     = model.predict(img_array)
    predicted_class = tf.argmax(predictions, axis=1).numpy()[0]
    confidence      = float(np.max(predictions)) * 100
    return predicted_class, categories[predicted_class], confidence


def show_recommendations(predicted_label):
    """Tampilkan rekomendasi produk skincare berdasarkan label prediksi.

    Membaca treatment.csv dan menampilkan daftar produk beserta gambarnya
    yang cocok dengan label kondisi kulit yang diprediksi.
    """
    df     = pd.read_csv('skincare_product/treatment.csv')
    produk = df[df['Tags'].str.lower() == predicted_label.lower()]

    if produk.empty:
        print(f"Tidak ada rekomendasi produk untuk kategori '{predicted_label}'.")
        return

    print(f"\nRekomendasi produk untuk '{predicted_label}':")
    for _, row in produk.iterrows():
        print(f"- {row['Brand']} | {row['Product Name']} | {row['Price']}")
        print(f"  Link: {row['Links']}")

        # Coba semua kemungkinan ekstensi gambar produk
        img_found = False
        for ext in ['png', 'jpg', 'jpeg', 'webp']:
            img_path = f"skincare_product/gambar_produk/{row['Id']}.{ext}"
            if os.path.exists(img_path):
                img_prod = image.load_img(img_path, target_size=(720, 1280))
                plt.figure()
                plt.imshow(img_prod)
                plt.title(f"{row['Brand']} - {row['Product Name']}")
                plt.axis('off')
                plt.show()
                img_found = True
                break

        if not img_found:
            # Fallback: glob untuk menangkap nama file yang tidak standar
            matches = glob.glob(f"skincare_product/gambar_produk/{row['Id']}.*")
            if matches:
                img_prod = image.load_img(matches[0], target_size=(720, 1280))
                plt.figure()
                plt.imshow(img_prod)
                plt.title(f"{row['Brand']} - {row['Product Name']}")
                plt.axis('off')
                plt.show()
            else:
                print(f"  [!] Gambar produk tidak ditemukan untuk ID: {row['Id']}")


# %% 15. Contoh Prediksi pada Gambar Baru

img_path = "darkspot.jpg"   # ganti dengan path gambar yang ingin diuji

if os.path.exists(img_path):
    predicted_class_index, predicted_class_name, confidence = predict_skin_condition(
        model, img_path, categories
    )
    print(f"Predicted class index : {predicted_class_index}")
    print(f"Predicted class name  : {predicted_class_name}")
    print(f"Confidence            : {confidence:.2f}%")

    img_disp = image.load_img(img_path, target_size=(720, 1280))
    plt.imshow(img_disp)
    plt.title(f"Predicted: {predicted_class_name} ({confidence:.2f}%)")
    plt.axis('off')
    plt.show()

    show_recommendations(predicted_class_name)
else:
    print(f"[!] File '{img_path}' tidak ditemukan. Ganti dengan path gambar yang valid.")


# %% 16. Evaluasi — Confusion Matrix & Classification Report

print("\nEvaluasi pada validation set...")

# Kumpulkan label prediksi dari seluruh batch validasi
y_true = np.array(y_val)
y_pred = []
for batch_imgs, _ in val_ds:
    probs      = model.predict(batch_imgs, verbose=0)
    batch_pred = np.argmax(probs, axis=1)
    y_pred.extend(batch_pred)

# Potong jika jumlah prediksi melebihi jumlah label asli (akibat batching)
y_pred = np.array(y_pred[:len(y_true)])

# Confusion matrix
cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
print("Confusion Matrix:\n", cm)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=categories, yticklabels=categories)
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.title('Confusion Matrix (Validation)')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.show()
print("Confusion matrix disimpan ke 'confusion_matrix.png'.")

# Classification report
report = classification_report(y_true, y_pred, target_names=categories, digits=4)
print("\nClassification Report:\n", report)
with open('classification_report.txt', 'w', encoding='utf-8') as f:
    f.write(report)
print("Classification report disimpan ke 'classification_report.txt'.")


# %% 17. Simpan Model

BEST_MODEL_PATH  = 'best_skin_model.h5'   # disimpan otomatis oleh ModelCheckpoint Stage 2
FINAL_MODEL_PATH = 'final_skin_model.h5'  # snapshot model setelah epoch terakhir

# Simpan snapshot akhir (tidak menimpa bobot terbaik dari checkpoint)
model.save(FINAL_MODEL_PATH)
print(f"Model akhir disimpan ke '{FINAL_MODEL_PATH}'.")


# %% 18. Ekspor Model ke ONNX

import tf2onnx
import onnx

# Muat dari checkpoint Stage 2 (bobot terbaik berdasarkan val_accuracy)
export_model = tf.keras.models.load_model(BEST_MODEL_PATH)

# Tentukan signature input dengan resolusi tetap 320×320 (sesuai Stage 2)
input_name = export_model.inputs[0].name.split(':')[0]
spec = (tf.TensorSpec((None, IMG_SIZE_STAGE2, IMG_SIZE_STAGE2, 3), tf.float32, name=input_name),)

# Opset 15 kompatibel dengan onnxruntime modern dan TensorFlow 2.x
output_path = 'best_skin_model_320.onnx'
model_proto, _ = tf2onnx.convert.from_keras(
    export_model,
    input_signature=spec,
    opset=15,
    output_path=output_path,
)
onnx.save_model(model_proto, output_path)
print(f"Model berhasil diekspor ke '{output_path}'.")
