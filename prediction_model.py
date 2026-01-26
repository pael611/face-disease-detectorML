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

# %%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report
import glob


# %%
# Optional: auto-crop to face region to reduce background noise.
# Uses OpenCV Haar Cascade if available; falls back to the original behavior if not.
ENABLE_AUTO_FACE_CROP = True
FACE_CROP_MARGIN = 0.25  # expand detected face box by 25% on each side

# Debug: save preview images of face-crop results to disk.
# This is useful to verify cropping quality before running long trainings.
DEBUG_FACE_CROP_PREVIEW = False
DEBUG_FACE_CROP_DIR = 'debug_face_crop'
DEBUG_FACE_CROP_MAX_SAVES = 40

_FACE_CASCADE = None

_DEBUG_FACE_CROP_SAVED = 0


def _maybe_save_face_crop_preview(
    *,
    source_path: str,
    before_img,
    cropped_img,
    final_img,
    target_img_size: int,
    resample,
):
    """Save a side-by-side preview (before | cropped | final) if debug is enabled."""
    global _DEBUG_FACE_CROP_SAVED

    if not DEBUG_FACE_CROP_PREVIEW:
        return

    # Avoid heavy I/O: only save the first N samples.
    try:
        if _DEBUG_FACE_CROP_SAVED >= DEBUG_FACE_CROP_MAX_SAVES:
            return
        idx = _DEBUG_FACE_CROP_SAVED
        _DEBUG_FACE_CROP_SAVED += 1

        os.makedirs(DEBUG_FACE_CROP_DIR, exist_ok=True)

        from PIL import Image, ImageOps
        import hashlib

        stem = os.path.basename(source_path)
        h = hashlib.md5(source_path.encode('utf-8', errors='ignore')).hexdigest()[:8]
        out_name = f"{idx:04d}_s{target_img_size}_{stem}_{h}.jpg"
        out_path = os.path.join(DEBUG_FACE_CROP_DIR, out_name)

        # Normalize sizes so comparison is meaningful
        a = ImageOps.pad(before_img, (target_img_size, target_img_size), method=resample)
        b = ImageOps.pad(cropped_img, (target_img_size, target_img_size), method=resample)
        c = final_img

        grid = Image.new('RGB', (target_img_size * 3, target_img_size))
        grid.paste(a, (0, 0))
        grid.paste(b, (target_img_size, 0))
        grid.paste(c, (target_img_size * 2, 0))
        grid.save(out_path, quality=92)
    except Exception:
        # Debug preview must never break training/inference
        return


def _get_face_cascade():
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
    """Crop PIL image to the largest detected face (with margin). If detection fails, return original img."""
    cascade = _get_face_cascade()
    if cascade is None:
        return img

    try:
        import cv2
        import numpy as np

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

        # Choose the largest face
        x, y, fw, fh = max(faces, key=lambda b: b[2] * b[3])
        mx = int(fw * margin)
        my = int(fh * margin)
        x1 = max(0, x - mx)
        y1 = max(0, y - my)
        x2 = min(w, x + fw + mx)
        y2 = min(h, y + fh + my)

        if x2 <= x1 or y2 <= y1:
            return img
        return img.crop((x1, y1, x2, y2))
    except Exception:
        return img


# %%
dataset_path = 'dataset'
categories = ['Acne', 'Blackheads', 'Dark Spots', 'Normal Skin', 'Oily Skin', 'Wrinkles']
# Two-stage training:
# - Stage 1: warmup classifier head on 224 for speed/stability
# - Stage 2: fine-tune on 320 to preserve more detail
IMG_SIZE_STAGE1 = 224
BATCH_SIZE_STAGE1 = 16
IMG_SIZE_STAGE2 = 320
BATCH_SIZE_STAGE2 = 8

# Keep final (inference/export) settings aligned to Stage 2
img_size = IMG_SIZE_STAGE2
BATCH_SIZE = BATCH_SIZE_STAGE2
num_classes = len(categories)

# %%
datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2,
    rotation_range=10,
    width_shift_range=0.05,
    height_shift_range=0.05,
    horizontal_flip=True,
    brightness_range=[0.9, 1.1],
    zoom_range=0.1,
    fill_mode='reflect',
    interpolation_order=2
)

train_gen = datagen.flow_from_directory(
    dataset_path,
    target_size=(img_size, img_size),
    batch_size=BATCH_SIZE,
    class_mode='sparse',
    subset='training',
    shuffle=True
)

val_gen = datagen.flow_from_directory(
    dataset_path,
    target_size=(img_size, img_size),
    batch_size=BATCH_SIZE,
    class_mode='sparse',
    subset='validation',
    shuffle=False
)

# %%
print("Mapping folder ke label (class_indices):", train_gen.class_indices)
print("Urutan categories:", categories)
print("Distribusi label di data training:", np.bincount(train_gen.classes))

# %%
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_gen.classes),
    y=train_gen.classes
)
class_weights = dict(enumerate(class_weights))
print("Class weights:", class_weights)

# %%
# 15. Siapkan daftar file & label, split stratified
import os, glob
from sklearn.model_selection import train_test_split

# Gunakan variabel yang sudah ada: dataset_path, categories, img_size, BATCH_SIZE
# Hanya gunakan format yang didukung decoder TensorFlow: JPEG, PNG, BMP, GIF
image_exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.gif")

filepaths = []
labels = []
for cls_idx, cls_name in enumerate(categories):
    cls_dir = os.path.join(dataset_path, cls_name)
    cls_files = []
    for ext in image_exts:
        cls_files.extend(glob.glob(os.path.join(cls_dir, ext)))
    # Tambahkan ke list global
    filepaths += cls_files
    labels += [cls_idx] * len(cls_files)

# Filter defensif: pastikan ekstensi valid (hindari file non-gambar)
valid_suffix = (".jpg", ".jpeg", ".png", ".bmp", ".gif")
filtered = [(fp, lab) for fp, lab in zip(filepaths, labels) if fp.lower().endswith(valid_suffix)]
filepaths = [fp for fp, _ in filtered]
labels = [lab for _, lab in filtered]

# Split stratified untuk train/valid (20% validasi)
X_train, X_val, y_train, y_val = train_test_split(
    filepaths, labels, test_size=0.2, random_state=42, stratify=labels
)

# Tampilkan ringkas distribusi sebelum balancing
import numpy as np
unique, counts = np.unique(y_train, return_counts=True)
print("Distribusi train sebelum balancing:")
for u, c in zip(unique, counts):
    print(f"  {categories[u]}: {c}")

# %%
# 16. Bangun tf.data untuk oversampling per-kelas (robust decoding via PIL)
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image, ImageOps
import numpy as np

AUTOTUNE = tf.data.AUTOTUNE

def _make_data_augmentation(name: str = 'data_augmentation'):
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
    data_augmentation = _make_data_augmentation(f'data_augmentation_{target_img_size}')

    def pil_decode_and_resize(path):
        path_str = path.numpy().decode('utf-8')
        try:
            resample = Image.Resampling.BICUBIC
        except AttributeError:
            resample = Image.BICUBIC
        with Image.open(path_str) as img:
            img = ImageOps.exif_transpose(img)
            img = img.convert('RGB')

            before = img

            if ENABLE_AUTO_FACE_CROP:
                img = _auto_face_crop_pil(img, margin=FACE_CROP_MARGIN)

            cropped = img

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

            arr = np.asarray(img, dtype=np.float32)
            return arr

    def _load_and_preprocess(path, label, training: bool):
        img = tf.py_function(func=pil_decode_and_resize, inp=[path], Tout=tf.float32)
        img = tf.reshape(img, [target_img_size, target_img_size, 3])

        if training and apply_augment:
            img01 = tf.clip_by_value(img / 255.0, 0.0, 1.0)
            img01 = data_augmentation(img01, training=True)
            img = tf.clip_by_value(img01 * 255.0, 0.0, 255.0)

        img = preprocess_input(img)
        return img, label

    def _load_and_preprocess_train(path, label):
        return _load_and_preprocess(path, label, training=True)

    def _load_and_preprocess_val(path, label):
        return _load_and_preprocess(path, label, training=False)

    return _load_and_preprocess_train, _load_and_preprocess_val

def build_balanced_datasets(target_img_size: int, batch_size: int):
    _load_and_preprocess_train, _load_and_preprocess_val = _make_preprocess_fns(
        target_img_size=target_img_size,
        apply_augment=True,
    )

    # Validation dataset (tanpa augment)
    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    val_ds = val_ds.map(_load_and_preprocess_val, num_parallel_calls=AUTOTUNE)
    val_ds = val_ds.batch(batch_size).prefetch(AUTOTUNE)

    # Train dataset (oversampling)
    per_class_datasets = []
    class_counts = []
    for cls_idx, cls_name in enumerate(categories):
        cls_files = [fp for fp, lab in zip(X_train, y_train) if lab == cls_idx]
        class_counts.append(len(cls_files))
        ds = tf.data.Dataset.from_tensor_slices((cls_files, [cls_idx] * len(cls_files)))
        ds = ds.shuffle(max(8 * batch_size, len(cls_files)))
        ds = ds.repeat()
        ds = ds.map(_load_and_preprocess_train, num_parallel_calls=AUTOTUNE)
        per_class_datasets.append(ds)

    num_classes_local = len(categories)
    weights = [1.0 / num_classes_local] * num_classes_local
    balanced_ds = tf.data.experimental.sample_from_datasets(per_class_datasets, weights=weights)
    train_ds_balanced = balanced_ds.batch(batch_size).prefetch(AUTOTUNE)

    steps_per_epoch = max(1, sum(class_counts) // batch_size)
    return train_ds_balanced, val_ds, class_counts, steps_per_epoch

print("Oversampling siap. Decoder: PIL (detail-preserving).")

# %% [markdown]
# <h1>Membangun Model</h1>

# %%
#Transfer Learning Model (MobileNetV2 + Decision Layer)
base_model = MobileNetV2(
    input_shape=(None, None, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

# %%
inputs = Input(shape=(None, None, 3))
x = base_model(inputs, training=False)
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
outputs = Dense(num_classes, activation='softmax')(x)
model = Model(inputs, outputs)

# %%
# 7. Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)
checkpoint = ModelCheckpoint(
    'best_skin_model.h5',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

# %%
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# %%
model.summary()

# %%
# 17. Two-stage training

# Stage 1: warmup on 224 (frozen backbone)
print(f"\nStage 1: warmup @ {IMG_SIZE_STAGE1}x{IMG_SIZE_STAGE1}, batch={BATCH_SIZE_STAGE1}")
train_ds_stage1, val_ds_stage1, class_counts_stage1, steps_per_epoch_stage1 = build_balanced_datasets(
    target_img_size=IMG_SIZE_STAGE1,
    batch_size=BATCH_SIZE_STAGE1,
)
print("steps_per_epoch_stage1:", steps_per_epoch_stage1)

checkpoint_stage1 = ModelCheckpoint(
    'best_skin_model_stage1.h5',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1,
)

history_stage1 = model.fit(
    train_ds_stage1,
    steps_per_epoch=steps_per_epoch_stage1,
    validation_data=val_ds_stage1,
    epochs=6,
    callbacks=[EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True), checkpoint_stage1],
)

# Stage 2: fine-tune on 320 (batch 8)
print(f"\nStage 2: fine-tune @ {IMG_SIZE_STAGE2}x{IMG_SIZE_STAGE2}, batch={BATCH_SIZE_STAGE2}")
train_ds_balanced, val_ds, class_counts, steps_per_epoch = build_balanced_datasets(
    target_img_size=IMG_SIZE_STAGE2,
    batch_size=BATCH_SIZE_STAGE2,
)
print("steps_per_epoch_stage2:", steps_per_epoch)

base_model.trainable = True
for layer in base_model.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
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

# Plot training curves (accuracy & loss) for balanced training
def plot_training_history(history):
    acc = history.history.get('accuracy', [])
    val_acc = history.history.get('val_accuracy', [])
    loss = history.history.get('loss', [])
    val_loss = history.history.get('val_loss', [])

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Train Accuracy')
    plt.plot(val_acc, label='Val Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Train Loss')
    plt.plot(val_loss, label='Val Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()

plot_training_history(history_balanced)

# %%
from tensorflow.keras.preprocessing import image

def load_and_preprocess_image(img_path, target_size=None):
    if target_size is None:
        target_size = (img_size, img_size)
    try:
        from PIL import Image, ImageOps

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
        # Fallback to Keras loader if PIL path fails
        img = image.load_img(img_path, target_size=target_size)
        img_array = image.img_to_array(img)

    img_array = preprocess_input(img_array)
    img_array = tf.expand_dims(img_array, axis=0)
    return img_array

def predict_skin_condition(model, img_path, categories):
    img_array = load_and_preprocess_image(img_path)
    predictions = model.predict(img_array)
    predicted_class = tf.argmax(predictions, axis=1).numpy()[0]
    confidence = float(np.max(predictions)) * 100
    return predicted_class, categories[predicted_class], confidence

def show_recommendations(predicted_label):
    df = pd.read_csv('skincare_product/treatment.csv')
    produk = df[df['Tags'].str.lower() == predicted_label.lower()]
    if produk.empty:
        print("Tidak ada rekomendasi produk untuk kategori ini.")
        return
    print(f"\nRekomendasi produk untuk '{predicted_label}':")
    for _, row in produk.iterrows():
        print(f"- {row['Brand']} | {row['Product Name']} | {row['Price']}")
        print(f"  Link: {row['Links']}")
        # Cek semua kemungkinan ekstensi gambar
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
            # Jika tidak ditemukan, coba cari dengan glob (jaga-jaga ada nama file aneh)
            pattern = f"skincare_product/gambar_produk/{row['Id']}.*"
            matches = glob.glob(pattern)
            if matches:
                img_prod = image.load_img(matches[0], target_size=(720, 1280))
                plt.figure()
                plt.imshow(img_prod)
                plt.title(f"{row['Brand']} - {row['Product Name']}")
                plt.axis('off')
                plt.show()
            else:
                print(f"  Gambar produk tidak ditemukan untuk ID: {row['Id']}")

# Contoh penggunaan prediksi gambar baru
img_path = "darkspot.jpg"  # Ganti dengan path gambar yang ingin diuji
if os.path.exists(img_path):
    predicted_class_index, predicted_class_name, confidence = predict_skin_condition(model, img_path, categories)
    print(f"Predicted class index: {predicted_class_index}")
    print(f"Predicted class name: {predicted_class_name}")
    print(f"Confidence: {confidence:.2f}%")
    img_disp = image.load_img(img_path, target_size=(720, 1280))
    plt.imshow(img_disp)
    plt.title(f"Predicted: {predicted_class_name} ({confidence:.2f}%)")
    plt.axis('off')
    plt.show()
    show_recommendations(predicted_class_name)
else:
    print(f"File {img_path} tidak ditemukan.")

# %%
# 18. Evaluasi: Confusion Matrix & Classification Report pada validation set
print("\nEvaluasi pada validation set (confusion matrix & classification report)...")

# Kumpulkan prediksi pada val_ds
y_true = np.array(y_val)
y_pred = []
for batch_imgs, _ in val_ds:
    probs = model.predict(batch_imgs, verbose=0)
    batch_pred = np.argmax(probs, axis=1)
    y_pred.extend(batch_pred)
y_pred = np.array(y_pred[:len(y_true)])  # jaga-jaga jika overshoot

# Confusion matrix
cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
print("Confusion Matrix:\n", cm)

# Plot heatmap
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

# %%
# 13. SIMPAN MODEL
BEST_MODEL_PATH = 'best_skin_model.h5'  # saved by ModelCheckpoint during Stage 2
FINAL_MODEL_PATH = 'final_skin_model.h5'

# Simpan model hasil akhir training (opsional) tanpa menimpa model terbaik dari checkpoint
model.save(FINAL_MODEL_PATH)

# %%
#14.EKSPOR MODEL KE ONNX
import tf2onnx
import onnx

# Ekspor dari model TERBAIK (checkpoint stage-2) agar sesuai dengan hasil terbaik validasi
export_model = tf.keras.models.load_model(BEST_MODEL_PATH)

# Pakai input fixed 320x320 agar konsisten dengan pipeline Stage 2
input_name = export_model.inputs[0].name.split(':')[0]
spec = (tf.TensorSpec((None, IMG_SIZE_STAGE2, IMG_SIZE_STAGE2, 3), tf.float32, name=input_name),)

# Opset 15 umumnya aman untuk ONNX/onnxruntime modern dan cocok dengan ekosistem TF2.10
output_path = 'best_skin_model_320.onnx'
model_proto, _ = tf2onnx.convert.from_keras(
    export_model,
    input_signature=spec,
    opset=15,
    output_path=output_path,
)
onnx.save_model(model_proto, output_path)
print(f"Model berhasil diekspor ke {output_path}")

# %%
