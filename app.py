from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os
app = Flask(__name__)
model = tf.keras.models.load_model('cats_vs_dogs_vgg16.h5')

# class indices
class_names = ['Cat', 'Dog']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return 'No image uploaded', 400

    file = request.files['image']
    if file.filename == '':
        return 'No selected file', 400

    img_path = os.path.join('static', file.filename)
    file.save(img_path)

    img = image.load_img(img_path, target_size=(224, 224))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor = img_tensor / 255.0

    prediction = model.predict(img_tensor)[0][0]
    label = class_names[1] if prediction > 0.5 else class_names[0]
    confidence = prediction if prediction > 0.5 else 1 - prediction

    return render_template('result.html', label=label, confidence=round(confidence * 100, 2), user_image=img_path)

if __name__ == '__main__':
    app.run(debug=True)
