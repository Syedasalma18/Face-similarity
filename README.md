# Face Similarity

This project compares the similarity between two faces using facial embeddings. It uses deep learning models to extract face features and compute similarity scores.

## 🚀 Features

- Face detection using MTCNN or OpenCV
- Face embeddings with pre-trained models (e.g., FaceNet)
- Cosine similarity or Euclidean distance scoring
- Streamlit interface (if included) or command-line script

## 📁 Project Structure

```
Face-similarity-main/
├── data/                # Folder to store test images
├── models/              # Pre-trained face recognition models
├── face_similarity.py   # Core script to compare faces
├── utils.py             # Helper functions for preprocessing and evaluation
├── requirements.txt     # Required Python libraries
└── README.md            # Project documentation
```

## 🛠️ Installation

```bash
git clone https://github.com/yourusername/Face-similarity.git
cd Face-similarity-main
pip install -r requirements.txt
```

## 🖼️ How to Use

### Command Line

```bash
python face_similarity.py --img1 path/to/image1.jpg --img2 path/to/image2.jpg
```

### Streamlit App (if included)

```bash
streamlit run app.py
```

## 🧪 Requirements

- Python 3.7+
- TensorFlow or PyTorch
- OpenCV
- NumPy
- scikit-learn
- MTCNN or dlib (for face detection)

Install dependencies:

```bash
pip install -r requirements.txt
```

## 📊 Output

- Similarity score between two faces
- Visual comparison of the two faces (optional)

## 👨‍💻 Author

Syeda Salma Syed Arif  
Master's in Cybersecurity, Drexel University


