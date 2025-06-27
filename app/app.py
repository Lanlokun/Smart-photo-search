from flask import Flask, request, render_template, redirect, send_from_directory, jsonify, flash
from werkzeug.utils import secure_filename
from app.yolo import detect_objects
from app.models import Photo, db
from app.search import preload_image_embeddings, get_image_embedding, build_faiss_index, fast_search
from dotenv import load_dotenv
from flask_migrate import Migrate
import os
import math
from datetime import datetime

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY')

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'Uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///photos.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)
migrate = Migrate(app, db)

# Global cache for semantic search
image_embeddings_cache = {}
faiss_index = None
photo_list = []

with app.app_context():
    db.create_all()
    all_photos = Photo.query.all()
    image_embeddings_cache = preload_image_embeddings(all_photos)
    faiss_index, photo_list = build_faiss_index(image_embeddings_cache)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_tags_from_filename(filename):
    name = os.path.splitext(os.path.basename(filename))[0]
    parts = name.split('_')[:-1]
    return [(tag, 1.0) for tag in parts if tag]

@app.route('/', methods=['GET', 'POST'])
def index():
    global image_embeddings_cache, faiss_index, photo_list

    query = request.args.get('query', '').strip().lower()
    page = request.args.get('page', 1, type=int)
    IMAGES_PER_PAGE = 12

    if request.method == 'POST':
        files = request.files.getlist('file')
        invalid_files = []
        saved_files = []

        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)

                labels = detect_objects(filepath)
                label_str = "_".join(labels) if labels else "nolabel"
                new_filename = f"{label_str}_{filename}"
                new_path = os.path.join(app.config['UPLOAD_FOLDER'], new_filename)
                os.rename(filepath, new_path)

                mtime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                embedding_vector = get_image_embedding(new_path)
                Photo.add_photo(new_filename, labels, mtime, embedding_vector)

                photo_obj = Photo.query.filter_by(filename=new_filename).first()
                if photo_obj:
                    image_embeddings_cache[photo_obj] = embedding_vector

                saved_files.append(new_filename)
            else:
                invalid_files.append(file.filename)

        if invalid_files:
            return jsonify({
                "success": False,
                "message": f"Invalid file type(s): {', '.join(invalid_files)}. Allowed types are: {', '.join(ALLOWED_EXTENSIONS)}"
            }), 400

        # Rebuild FAISS index after upload
        faiss_index, photo_list = build_faiss_index(image_embeddings_cache)

        return jsonify({
            "success": True,
            "message": f"Uploaded {len(saved_files)} file(s) successfully!"
        })

    if query:
        results = fast_search(query, faiss_index, photo_list)
        total_images = len(results)
        start = (page - 1) * IMAGES_PER_PAGE
        end = start + IMAGES_PER_PAGE
        photos = [photo for photo, score in results[start:end]]
    else:
        photos, total_images = Photo.get_paginated_photos(query, page, IMAGES_PER_PAGE)

    total_pages = max(1, math.ceil(total_images / IMAGES_PER_PAGE))

    images_with_tags = []
    for photo in photos:
        tags = [(tag, 1.0) for tag in photo.tags.split(',') if tag]
        img_url = f'/Uploads/{photo.filename}'
        images_with_tags.append((img_url, tags, photo.mtime))

    return render_template(
        'index.html',
        images=images_with_tags,
        page=page,
        total_pages=total_pages,
        total_images=total_images,
        query=query,
        yolo_labels=[]
    )

@app.route('/Uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/delete/<filename>', methods=['POST'])
def delete_image(filename):
    global image_embeddings_cache, faiss_index, photo_list

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if os.path.exists(filepath):
        os.remove(filepath)

    photo = Photo.query.filter_by(filename=filename).first()
    if photo:
        image_embeddings_cache.pop(photo, None)
        db.session.delete(photo)
        db.session.commit()

    faiss_index, photo_list = build_faiss_index(image_embeddings_cache)

    return redirect('/')

@app.route('/search_suggestions')
def search_suggestions():
    query = request.args.get('q', '').lower()
    suggestions = set()
    for photo in Photo.query.all():
        for tag in photo.tags.split(','):
            if tag.lower().startswith(query):
                suggestions.add(tag.lower())
    return jsonify(list(suggestions)[:10])

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(host='0.0.0.0', port=5000, debug=True)
