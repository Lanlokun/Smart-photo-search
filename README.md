# Smart Photo Search

Smart Photo Search is a Flask web application that combines YOLO object detection and CLIP-based semantic embeddings to provide fast, AI-powered photo search using natural language queries. Upload images, get AI-generated tags, and find photos instantly with a user-friendly interface.

## Features

- Upload photos with automatic YOLO-based object detection and tagging
- Generate semantic embeddings using the CLIP model (clip-ViT-B-32)
- Fast similarity search powered by FAISS
- Natural language search to find photos by description or tags
- Responsive UI styled with Tailwind CSS
- Image preview, delete, and pagination support
- Search suggestions with autocomplete

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/smart-photo-search.git
   cd smart-photo-search
   ```
   
2. Create and activate a virtual environment:
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Set environment variables:

    Create a .env file in the project root

    Add FLASK_SECRET_KEY=your_secret_key_here

5. Initialize the database:

       ```bash
        flask db upgrade
        ```


## Usage

Run the application locally:
    ```bash
        flask run
        ```


- Visit http://localhost:5000 in your browser to start uploading and searching photos.



### Deployment
This app can be deployed on cloud platforms such as Render or Heroku. Ensure environment variables and database are configured accordingly.

### License
MIT License



Built with YOLO, CLIP, FAISS, Flask, and Tailwind CSS.

## Author 

- Mr. Malik Kolawole Lanlokun

- MSc Software Engineering Student

- Nankai University


