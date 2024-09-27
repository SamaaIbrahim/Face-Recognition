import dlib
import cv2
import numpy as np
import base64
from io import BytesIO
from flask import Flask, request, jsonify, render_template,url_for,redirect
from PIL import Image
import os
import glob
"""
class facerec:
    def __init__(self):
        # Load the pre-trained face detector and shape predictor model
        self.detector = dlib.get_frontal_face_detector()
        self.sp = dlib.shape_predictor(dlib.download_dlib_shape_predictor())
        self.face_recognition_model = dlib.face_recognition_model_v1(dlib.download_dlib_face_recognition_model())
        self.known_face_encodings = []
        self.known_face_names = []
        self.frame_resize = 0.25

    def encodings_imgs(self, images_path):
        images_path = glob.glob(os.path.join(images_path, "*.*"))
        for path in images_path:
            img = cv2.imread(path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            faces = self.detector(img_rgb)
            for face in faces:
                shape = self.sp(img_rgb, face)
                face_encoding = np.array(self.face_recognition_model.compute_face_descriptor(img_rgb, shape))
                self.known_face_encodings.append(face_encoding)
                basename = os.path.basename(path)
                (filename, ext) = os.path.splitext(basename)
                self.known_face_names.append(filename)

    def detect(self, frame):
        sm_frame = cv2.resize(frame, (0, 0), fx=self.frame_resize, fy=self.frame_resize)
        rgb_frame = cv2.cvtColor(sm_frame, cv2.COLOR_BGR2RGB)
        faces = self.detector(rgb_frame)
        face_encodings = []
        face_names = []
        for face in faces:
            shape = self.sp(rgb_frame, face)
            encoding = np.array(self.face_recognition_model.compute_face_descriptor(rgb_frame, shape))
            face_encodings.append(encoding)
            matches = [np.linalg.norm(encoding - known_encoding) < 0.6 for known_encoding in self.known_face_encodings]
            if True in matches:
                match_index = matches.index(True)
                name = self.known_face_names[match_index]
            else:
                name = "Unknown"
            face_names.append(name)
        face_locations = [(face.left(), face.top(), face.right(), face.bottom()) for face in faces]
        face_locations = np.array(face_locations) / self.frame_resize
        return face_locations.astype(int), face_names

    def detect_face(self, image):
        rgb_image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
        faces = self.detector(rgb_image)
        return len(faces) > 0

face_recognition_service = facerec()
face_recognition_service.encodings_imgs('images')
"""

IMAGE_DIR = 'static/images'
if not os.path.exists(IMAGE_DIR):
    os.makedirs(IMAGE_DIR)

action=''
app = Flask(__name__)
new_password = "1234"
@app.route('/')
def index():
    return render_template('faceRec.html')


# Directory to save images
IMAGE_DIR = 'static/images'
if not os.path.exists(IMAGE_DIR):
    os.makedirs(IMAGE_DIR)

@app.route('/homeRes', methods=['POST'])
def homeRes():
    try:
        data = request.get_json()
        img_data = data.get('image')

        if img_data:
            # Process the base64-encoded image
            img_data = img_data.replace('data:image/png;base64,', '')
            img_bytes = base64.b64decode(img_data)
            image = Image.open(BytesIO(img_bytes))
             # Convert image to RGB if it is in RGBA mode
            if image.mode == 'RGBA':
                image = image.convert('RGB')
            # Save the image with a default name
            file_path = os.path.join(IMAGE_DIR, 'unknown_face.jpg')
            image.save(file_path)

            # Return URL to the image for rendering
            return jsonify({'redirect_url': url_for('homeOwners', img_path='images/unknown_face.jpg')})

        return jsonify({'error': 'No image data provided'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/homeowner')
def homeOwners():
    img_path = request.args.get('img_path', 'images/unknown_face.jpg')
    return render_template('owners.html', src=img_path)

@app.route('/decision-action', methods=['POST'])
def decisionAction():
    try:
        action = request.form.get('action')
        img_path = request.form.get('img_path')
        name = request.form.get('name', 'unknown')
        
        if action == 'open':
            # Here you would perform the action to open the door
            return jsonify({'message': 'Door opened!'})

        elif action == 'add_and_open':
            # Save the image with the given name
            new_path = os.path.join('images', f'{name}.jpg')
            os.rename(os.path.join(IMAGE_DIR, 'unknown_face.jpg'), new_path)
            # Here you would perform the action to open the door
            return jsonify({'message': f'Door opened and {name} added to the system!'})

        return jsonify({'error': 'Invalid action'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/check_password', methods=['POST'])
def check_password():
    data = request.get_json()
    password = data['password']
    if password == new_password:  
        return jsonify({'message': 'Password correct, proceed to add user.'})
    else:
        return jsonify({'message': 'Incorrect password.'}), 400
@app.route('/change_password', methods=['POST'])
def change_password():
    if request.method == 'POST':
        data = request.get_json()
        new_password = data['new_password']
        #connection.write(new_password.encode())  
        return jsonify({'status': 'Password updated', 'new_password': new_password})
   


@app.route('/recognize_face', methods=['POST'])
def recognize_face():
    if request.method == 'POST':
        data = request.get_json()
        img_data = data['image']
        img_data = img_data.replace('data:image/png;base64,', '')
        img_bytes = base64.b64decode(img_data)
        image = Image.open(BytesIO(img_bytes))
       # face_recognition_service.encodings_imgs('images')

        return jsonify({"name": 'Unknown', "face_locations": [[50, 69,90,100]]})
    elif request.method == 'GET':
        return jsonify({'message': 'Please send a POST request with a base64-encoded image to recognize faces.'})

@app.route('/add-user', methods=['POST'])
def add_user():
    name = request.form.get('name')
    img_data = request.files.get('image')

    if not img_data or not name:
        return jsonify({'error': 'Name and image are required.'}), 400
    
    # Save the image directly from the uploaded file
    file_path = os.path.join('images', f'{name}.jpg')
    
    # Use Pillow to handle the image if you need to process it
    image = Image.open(img_data)
    
    # Convert image to RGB if it is in RGBA mode
    if image.mode == 'RGBA':
        image = image.convert('RGB')

    image.save(file_path)

    return jsonify({'message': 'User added successfully.'})
  
@app.route('/delete_user', methods=['POST', 'GET'])
def delete_user():
    if request.method == 'POST':
        data = request.get_json()
        name = data.get('name') 
        
        if not name:
            return jsonify({'message': 'No name provided in request.'}), 400
        
        # List of possible image formats
        possible_extensions = ['.jpg', '.jpeg', '.png']
        file_deleted = False
        
        for ext in possible_extensions:
            file_path = os.path.join('images', f'{name}{ext}')
            if os.path.exists(file_path):
                os.remove(file_path)  # Remove user image file
                file_deleted = True
                break
        
        if file_deleted:

            return jsonify({'status': 'success', 'message': f'User {name} deleted successfully.'})
        else:
            return jsonify({'status': 'error', 'message': f'User {name} not found.'}), 404
    
    elif request.method == 'GET':
        return jsonify({'message': 'Please send a POST request to delete a user.'})
if __name__ == "__main__":
    app.run(debug=True)
