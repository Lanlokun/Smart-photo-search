from flask_sqlalchemy import SQLAlchemy
import json
import numpy as np

db = SQLAlchemy()

class Photo(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), unique=True, nullable=False)
    mtime = db.Column(db.String(100))
    tags = db.Column(db.String(500))  # Comma-separated
    embedding = db.Column(db.Text)  # JSON string of embedding vector

    @staticmethod
    def add_photo(filename, tags, mtime, embedding_vector=None):
        tag_str = ",".join(tags)
        # Ensure vector is normalized float32 list
        if embedding_vector is not None:
            embedding_vector = embedding_vector / np.linalg.norm(embedding_vector)
            embedding_json = json.dumps(embedding_vector.astype('float32').tolist())
        else:
            embedding_json = None

        photo = Photo(filename=filename, mtime=mtime, tags=tag_str, embedding=embedding_json)
        db.session.add(photo)
        db.session.commit()

    def get_embedding(self) -> np.ndarray:
        if self.embedding:
            emb = np.array(json.loads(self.embedding), dtype='float32')
            # Optional: Normalize again to be safe
            return emb / np.linalg.norm(emb)
        return None

    @staticmethod
    def get_paginated_photos(query, page, per_page):
        base_query = Photo.query
        if query:
            base_query = base_query.filter(Photo.tags.contains(query))
        base_query = base_query.order_by(Photo.mtime.desc())

        total = base_query.count()
        photos = base_query.offset((page - 1) * per_page).limit(per_page).all()
        return photos, total

    @staticmethod
    def delete_by_filename(filename):
        photo = Photo.query.filter_by(filename=filename).first()
        if photo:
            db.session.delete(photo)
            db.session.commit()

    @staticmethod
    def rebuild_from_folder(folder, allowed_exts, extract_tags_from_filename):
        import os
        from datetime import datetime
        for f in os.listdir(folder):
            if '.' in f and f.rsplit('.', 1)[1].lower() in allowed_exts:
                if not Photo.query.filter_by(filename=f).first():
                    path = os.path.join(folder, f)
                    tags = extract_tags_from_filename(f)
                    tag_str = ",".join([tag for tag, _ in tags])
                    mtime = datetime.fromtimestamp(os.path.getmtime(path)).strftime('%Y-%m-%d %H:%M:%S')
                    photo = Photo(filename=f, mtime=mtime, tags=tag_str)
                    db.session.add(photo)
        db.session.commit()

    @staticmethod
    def semantic_search(query: str, page: int, per_page: int, cache: dict):
        from app.search import search_images_by_description
        results = search_images_by_description(query, cache)

        total = len(results)
        start = (page - 1) * per_page
        end = start + per_page
        paginated = results[start:end]

        photos_paginated = [item[0] for item in paginated]
        return photos_paginated, total
