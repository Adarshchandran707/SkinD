# from flask import Flask, render_template, request, redirect
# from werkzeug.utils import secure_filename
# import os
# import numpy as np
# from keras.preprocessing import image
# import matplotlib.pyplot as plt
# from keras.models import load_model
# from keras.preprocessing.image import ImageDataGenerator
#
# app = Flask(__name__)
#
# UPLOAD_FOLDER = 'uploads'
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
#
#
# # Add your image classification model here
# # For example, assuming 'model' is your trained model
# # model = ...
#
# model_path = "C:/Users/adars/Downloads/inception.h5"
# loaded_model = load_model(model_path)
# test_datagen = ImageDataGenerator(rescale=1./255)
#
# test_generator = test_datagen.flow_from_directory(
#     'path_to_test_directory',
#     target_size=(224, 224),
#     batch_size=32,
#     class_mode='categorical'  # or 'binary' depending on your problem
# )
# def classify_image(file_path, train_generator=None):
#     # Load the uploaded image
#     img = image.load_img(file_path, target_size=(224, 224))
#     img_array = image.img_to_array(img)
#     img_array /= 255.0
#
#     # Add Gaussian Noise to the image
#     noisy_img_array = img_array + np.random.normal(loc=0, scale=0.1, size=img_array.shape)
#
#     # Clip the values to be within the valid range [0, 1]
#     noisy_img_array = np.clip(noisy_img_array, 0, 1)
#
#     # Display the original and denoised images (optional)
#     plt.figure(figsize=(8, 4))
#     plt.subplot(1, 2, 1)
#     plt.title('Original Image')
#     plt.imshow(img_array[0])
#     plt.axis('off')
#
#     plt.subplot(1, 2, 2)
#     plt.title('Noisy Image')
#     plt.imshow(noisy_img_array[0])
#     plt.axis('off')
#     plt.show()
#
#     # Predict the class of the denoised image
#     img_array = np.expand_dims(noisy_img_array, axis=0)
#     predictions = loaded_model.predict(img_array)
#     predicted_class = np.argmax(predictions)
#
#     # Decode the predictions (assuming you have a list of class labels)
#     class_labels = train_generator.class_indices
#     predicted_label = list(class_labels.keys())[predicted_class]
#
#     return predicted_label
#
# @app.route('/')
# def index():
#     return render_template('index.html')
#
# @app.route('/upload', methods=['POST'])
# def upload_file():
#     if 'file' not in request.files:
#         return redirect(request.url)
#
#     file = request.files['file']
#
#     if file.filename == '':
#         return redirect(request.url)
#
#     if file:
#         filename = secure_filename(file.filename)
#         file_path = os.path.join(os.getcwd(), 'uploads', filename)
#         file.save(file_path)
#
#         # Classify the uploaded image
#         predicted_label = classify_image(file_path)
#
#         # Pass the result to the result.html template
#         return render_template('result.html', file_path=file_path, predicted_label=predicted_label)
# if __name__ == '__main__':
#     app.run(debug=True)
from flask import Flask, render_template, request, redirect
from werkzeug.utils import secure_filename
import os
import numpy as np
from keras.preprocessing import image
import matplotlib.pyplot as plt
from keras.models import load_model

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load your pre-trained model
model_path = "C:/Users/adars/Downloads/inception.h5"
loaded_model = load_model(model_path)

# ImageDataGenerator for preprocessing test images
from keras.preprocessing.image import ImageDataGenerator

test_datagen = ImageDataGenerator(rescale=1./255)

def classify_image(file_path):
    # Load the uploaded image
    img = image.load_img(file_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array /= 255.0

    # Add Gaussian Noise to the image (optional)
    noisy_img_array = img_array + np.random.normal(loc=0, scale=0.1, size=img_array.shape)
    noisy_img_array = np.clip(noisy_img_array, 0, 1)

    # Display the original and noisy images (optional)
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(img_array[0])
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title('Noisy Image')
    plt.imshow(noisy_img_array[0])
    plt.axis('off')
    plt.show()

    # Preprocess the image using test_datagen
    img_array = np.expand_dims(img_array, axis=0)
    img_array = test_datagen.standardize(img_array)

    # Predict the class of the image
    predictions = loaded_model.predict(img_array)
    predicted_class = np.argmax(predictions)

    # Decode the predictions (replace with your class labels)
    class_labels = ['Acne and Rosacea Photos', 'Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions',
                    'Atopic Dermatitis Photos', 'Cellulitis Impetigo and other Bacterial Infections']  # Replace with actual class labels
    predicted_label = class_labels[predicted_class]

    return predicted_label

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        return redirect(request.url)

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Classify the uploaded image
        predicted_label = classify_image(file_path)

        # Pass the result to the result.html template
        return render_template('result.html', file_path=os.path.abspath(file_path), predicted_label=predicted_label)

if __name__ == '__main__':
    app.run(debug=True)
