import os
import secrets
from flask import Flask, render_template, request, redirect, url_for, session, jsonify, flash
from werkzeug.utils import secure_filename
from pymongo import MongoClient
import datetime
from bson.objectid import ObjectId

# Import Machine Learning Model Stubs
from models.image_model import detect_image_deepfake
from models.video_model import detect_video_deepfake
from models.audio_model import detect_audio_deepfake

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)

# Configuration for file uploads
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__name__)), 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4', 'avi', 'wav', 'mp3'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# MongoDB Configuration
# NOTE: Connects to local MongoDB by default. Can be updated to MongoDB Atlas.
MONGO_URI = os.environ.get('MONGO_URI', 'mongodb://localhost:27017/')
try:
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=2000)
    db = client['deepfake_db']
    users_collection = db['users']
    results_collection = db['results']
    # Quickly ping to check connection
    client.server_info()
    db_connected = True
except Exception as e:
    print(f"MongoDB not connected. Running in demo mode. Error: {e}")
    db_connected = False

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ================= Routes ================= #

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        
        if db_connected:
            existing_user = users_collection.find_one({'email': email})
            if existing_user:
                flash('Email already registered', 'error')
                return redirect(url_for('register'))
            
            users_collection.insert_one({
                'username': username,
                'email': email,
                'password': password, # In production, hash this!
                'created_at': datetime.datetime.utcnow()
            })
        flash('Registration successful! Please login.', 'success')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        if db_connected:
            user = users_collection.find_one({'email': email, 'password': password})
            if user:
                session['user_id'] = str(user['_id'])
                session['username'] = user['username']
                return redirect(url_for('dashboard'))
            else:
                flash('Invalid email or password', 'error')
        else:
            # Demo mode login
            session['user_id'] = 'demo_id'
            session['username'] = 'Demo User'
            return redirect(url_for('dashboard'))
            
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    session.pop('username', None)
    return redirect(url_for('index'))

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))
        
    history = []
    if db_connected:
        user_id = session['user_id']
        history= list(results_collection.find({'user_id': user_id}).sort('created_at', -1).limit(5))
        
    return render_template('dashboard.html', history=history)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401

    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
        
    file = request.files['file']
    media_type = request.form.get('media_type', 'image') # image, video, audio
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Call appropriate detection model
        result = {}
        try:
            if media_type == 'image':
                result = detect_image_deepfake(filepath)
            elif media_type == 'video':
                result = detect_video_deepfake(filepath)
            elif media_type == 'audio':
                result = detect_audio_deepfake(filepath)
            else:
                return jsonify({'error': 'Invalid media type'}), 400
                
            # Store result in MongoDB
            if db_connected:
                results_collection.insert_one({
                    'user_id': session['user_id'],
                    'filename': filename,
                    'media_type': media_type,
                    'prediction': result.get('label', 'UNKNOWN'),
                    'confidence': result.get('confidence', 0.0),
                    'created_at': datetime.datetime.utcnow()
                })
                
            return jsonify({
                'success': True,
                'filename': filename,
                'result': result
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
            
    return jsonify({'error': 'File type not allowed'}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)
