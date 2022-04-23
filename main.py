from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import keras
import numpy as np

model = keras.models.load_model('assets/models/keras_model.h5', compile=False)

lable_list = {
    0: 'Pepper__bell___Bacterial_spot',
    1: 'Pepper__bell___healthy',
    2: 'Potato___Early_blight',
    3: 'Potato___healthy',
    4: 'Potato___Late_blight',
    5: 'Tomato__Target_Spot',
    6: 'Tomato__Tomato_mosaic_virus',
    7: 'Tomato__Tomato_YellowLeaf__Curl_Virus',
    8: 'Tomato_Bacterial_spot',
    9: 'Tomato_Early_blight',
    10: 'Tomato_healthy',
    11: 'Tomato_Late_blight',
    12: 'Tomato_Leaf_Mold',
    13: 'Tomato_Septoria_leaf_spot',
    14: 'Tomato_Spider_mites_Two_spotted_spider_mite'
}
app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1


@app.route("/")
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    data = {}
    if request.method == 'POST':
        img = request.files['select_file']
        img.save('static/pic.jpg')
        image_path = "static/pic.jpg"
        data = predictImage(image_path)
        print(data)
    return (data), 200


def predictImage(image_path):
    data = tf.constant(image_path)
    # Read in image file
    image = tf.io.read_file(data)
    # Turn the jpeg image into numerical Tensor with 3 colour channels (Red, Green, Blue)
    image = tf.image.decode_jpeg(image, channels=3)
    # Convert the colour channel values from 0-225 values to 0-1 values
    image = tf.image.convert_image_dtype(image, tf.float32)
    # Resize the image to our desired size (224, 244)
    image = tf.image.resize(image, size=[224, 224])

    # reshaping to input size of the moedel
    image = tf.reshape(image, [1, 224, 224, 3])

    prdictedDisease = model.predict(image)
    print(prdictedDisease)
    itemindex = np.where(prdictedDisease == np.max(prdictedDisease))
    prob = np.max(prdictedDisease)
    label = lable_list[itemindex[1][0]]
    print(label)
    data = {'prod': str(prob), 'label': label}
    return data


if __name__ == "__main__":
    app.run(debug=True, host='127.0.0.1', port=5000)
