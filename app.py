from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import tensorflow as tf
import numpy as np
import PIL

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Set a secret key for session management

# Load the trained model
model = tf.keras.models.load_model('C:/Users/vijeth/FINAL PROJECT BCD Cancer/model.h5')

# Define the class names
class_names = ['benign', 'malignant']

# Define a function to preprocess the image
def preprocess_image(image):
    img = image.resize((180, 180))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Define a function to make predictions
def predict(image):
    preprocessed_image = preprocess_image(image)
    predictions = model.predict(preprocessed_image)
    score = tf.nn.softmax(predictions[0])
    class_index = np.argmax(score)
    class_name = class_names[class_index]
    confidence = 100 * np.max(score)
    return class_name, confidence

def is_user_authenticated():
    return 'username' in session

@app.route('/')
def home():
    if is_user_authenticated():
        return render_template('home.html')
    else:
        return redirect(url_for('login'))

@app.route('/check_patient')
def check_patient():
    if not is_user_authenticated():
        return redirect(url_for('login'))
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if is_user_authenticated():
        return redirect(url_for('home'))

    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Perform authentication
        if username == 'admin' and password == 'pwd':
            session['username'] = username  # Store the username in session
            return redirect(url_for('home'))
        else:
            error_message = 'Invalid username or password.'
            return render_template('login.html', error_message=error_message)

    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('username', None)  # Remove the username from session
    return redirect(url_for('login'))

@app.route('/predict', methods=['POST'])
def predict_image():
    # Check if an image file was uploaded
    if 'image' not in request.files:
        return render_template('index.html', message='No image file uploaded')

    image = request.files['image']

    # Check if the file is an image
    if image.filename == '':
        return render_template('index.html', message='No image file selected')

    try:
        img = PIL.Image.open(image)
        class_name, confidence = predict(img)
        return render_template('index.html', message='Prediction: {} (Confidence: {:.2f}%)'.format(class_name, confidence), class_name=class_name, confidence=confidence)
    except:
        return render_template('index.html', message='Error processing image')

        return render_template('index.html', message='Error processing image')


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
