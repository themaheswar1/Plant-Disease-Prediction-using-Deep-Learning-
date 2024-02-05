from __future__ import division, print_function
import matplotlib.pyplot as plt
import os
import numpy as np
# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image
# Flask utils
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = "C:/Users/thema/Desktop/flask api/FinalDraft.h5"

# Load your trained model
model = load_model(MODEL_PATH)

print('Model loaded here , you can Check http://127.0.0.1:5000/')


# Ensure the 'uploads' directory exists
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to temporary
        img_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename))
        f.save(img_path)
        img = image.load_img(img_path, target_size=(120, 120))

        plt.imshow(img)
        plt.title('Input Image')
        plt.show()

        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0

# Make prediction
        predictions = model.predict(img_array)
        print(predictions)
        class_indices = {'Tomato___Bacterial_spot': 0,
                          'Tomato___Early_blight': 1,
                         'Tomato___Late_blight': 2,
                           'Tomato___Leaf_Mold': 3,
                        'Tomato___Septoria_leaf_spot': 4,
                             'Tomato___Target_Spot': 5,
                       'Tomato___Tomato_Yellow_Leaf_Curl_Virus': 6,
                        'Tomato___Tomato_mosaic_virus': 7,
                     'Tomato___healthy': 8}

        class_labels = list(class_indices.keys())
# Map predictions to class labels
        predicted_class_index = np.argmax(predictions)
        predicted_class_label = class_labels[predicted_class_index]

# Show the predicted class label
        print(f'Predicted Class: {predicted_class_label}')
        return str(predicted_class_label)
    return None


if __name__ == '__main__':
    app.run(debug=True)

