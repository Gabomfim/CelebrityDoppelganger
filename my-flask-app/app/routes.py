from flask import Blueprint, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from PIL import Image
import io
from img2vec_pytorch import Img2Vec
import torch
import ssl
import certifi
import os
import numpy as np


# Configure SSL context
ssl._create_default_https_context = ssl._create_unverified_context

bp = Blueprint('main', __name__)

CELEBRITIES_FOLDER = 'app/celebrities/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def calculate_euclidean_distance(vec1, vec2):
    return np.linalg.norm(vec1 - vec2)

@bp.route('/')
def home():
    return render_template('base.html')

@bp.route('/about')
def about():
    return render_template('about.html')  # Ensure you create about.html in templates directory.

@bp.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            image = Image.open(io.BytesIO(file.read()))
            # Process the image here
            # For example, you can convert it to grayscale
            # Initialize Img2Vec
            # Remove the fourth channel if it exists
            if image.mode == 'RGBA':
                image = image.convert('RGB')

            img2vec = Img2Vec(cuda=torch.cuda.is_available())

            # Convert image to vector
            # Convert uploaded image to vector
            uploaded_image_vector = img2vec.get_vec(image)

            # Find the celebrity image with the shortest Euclidean distance
            shortest_distance = float('inf')
            closest_celebrity_image = None

            for filename in os.listdir(CELEBRITIES_FOLDER):
                if allowed_file(filename):
                    celebrity_image_path = os.path.join(CELEBRITIES_FOLDER, filename)
                    celebrity_image = Image.open(celebrity_image_path)
                    
                    # Remove the fourth channel if it exists
                    if celebrity_image.mode == 'RGBA':
                        celebrity_image = celebrity_image.convert('RGB')
                    
                    # Convert celebrity image to vector
                    celebrity_image_vector = img2vec.get_vec(celebrity_image)
                    
                    # Calculate Euclidean distance
                    distance = calculate_euclidean_distance(uploaded_image_vector, celebrity_image_vector)
                    
                    if distance < shortest_distance:
                        shortest_distance = distance
                        closest_celebrity_image = filename

            # Do something with the processed image
            # For now, we'll just return a success message
            print(f'Closest celebrity image: {closest_celebrity_image} with distance: {shortest_distance}')
            return f'Closest celebrity image: {closest_celebrity_image} with distance: {shortest_distance}'
    return render_template('upload.html')