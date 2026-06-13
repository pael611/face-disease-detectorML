# Pengujian model dengan berkas final_skin_model.h5
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
# Load model yang sudah dilatih
model = load_model('final_skin_model.h5')
# Fungsi untuk memprediksi kelas dari gambar input
def predict_skin_disease(img_path):
    img = image.load_img(img_path, target_size=(512, 512))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    return predicted_class 
# Contoh penggunaan.
categories   = ['Acne', 'Blackheads', 'Dark Spots', 'Normal Skin', 'Oily Skin', 'Wrinkles']
img_path = 'test dataset/Acne/09213c775f247e4ae9a706a42cb1fcb3.jpg'  # Ganti dengan path ke gambar yang ingin diuji
predicted_class = predict_skin_disease(img_path)
print(f'Predicted Class: {categories[predicted_class]}')
# visualisasi hasil prediksi dengan confidence score
import matplotlib.pyplot as plt
def visualize_prediction(img_path):
    img = image.load_img(img_path, target_size=(512, 512))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)[0]
    predicted_class = np.argmax(prediction)
    confidence_score = prediction[predicted_class]
    
    plt.imshow(image.load_img(img_path))
    plt.title(f'Predicted Class: {categories[predicted_class]}, Confidence: {confidence_score:.2f}')
    plt.axis('off')
    plt.show()
# Contoh penggunaan visualisasi
visualize_prediction(img_path)

