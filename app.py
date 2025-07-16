import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageEnhance
from deepface import DeepFace
import tempfile
import os
import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Streamlit page setup
st.set_page_config(page_title="Face Similarity Detector", layout="wide")

# Optimized preprocessing
def optimize_image(img):
    """Image optimization for better face detection"""
    try:
        # Convert to numpy array
        img_array = np.array(img)
        
        # Face detection and cropping
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # Face detection with relaxed parameters
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3, minSize=(50, 50))
        
        if len(faces) > 0:
            # Get the largest face
            face = max(faces, key=lambda x: x[2] * x[3])
            x, y, w, h = face
            
            # Small margin for context
            margin = int(0.1 * min(w, h))
            x = max(0, x - margin)
            y = max(0, y - margin)
            w = min(img_bgr.shape[1] - x, w + 2 * margin)
            h = min(img_bgr.shape[0] - y, h + 2 * margin)
            
            # Crop to face
            img_bgr = img_bgr[y:y+h, x:x+w]
            
            # Contrast enhancement only if needed
            if img_bgr.mean() < 100:
                img_bgr = cv2.convertScaleAbs(img_bgr, alpha=1.2, beta=20)
        
        # Ensure minimum size for good detection
        height, width = img_bgr.shape[:2]
        if width < 160 or height < 160:
            scale = max(160 / width, 160 / height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            img_bgr = cv2.resize(img_bgr, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        
        # Convert back to RGB
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img_rgb)
        
    except Exception as e:
        logger.warning(f"Optimization failed, using original: {e}")
        return img

# Similarity calculation with timeout
def calculate_similarity(img1, img2):
    """Similarity calculation using optimized model with timeout"""
    
    def run_analysis():
        try:
            # Image optimization
            img1_opt = optimize_image(img1)
            img2_opt = optimize_image(img2)
            
            # Create temporary files
            temp_dir = tempfile.gettempdir()
            
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False, dir=temp_dir) as tmp1, \
                 tempfile.NamedTemporaryFile(suffix=".jpg", delete=False, dir=temp_dir) as tmp2:
                
                # Save with optimized quality
                img1_opt.save(tmp1.name, format='JPEG', quality=85, optimize=True)
                img2_opt.save(tmp2.name, format='JPEG', quality=85, optimize=True)
                
                tmp1.close()
                tmp2.close()
                
                try:
                    # Use Facenet512 for balanced performance
                    result = DeepFace.verify(
                        img1_path=tmp1.name, 
                        img2_path=tmp2.name, 
                        model_name="Facenet512",
                        distance_metric="cosine",
                        enforce_detection=False,
                        silent=True
                    )
                    
                    similarity = (1 - result["distance"]) * 100
                    return similarity, result["verified"], None
                    
                finally:
                    # Cleanup
                    try:
                        os.unlink(tmp1.name)
                        os.unlink(tmp2.name)
                    except OSError:
                        pass
                        
        except Exception as e:
            return 0, False, str(e)
    
    # Run with timeout
    try:
        with ThreadPoolExecutor() as executor:
            future = executor.submit(run_analysis)
            return future.result(timeout=15)
    except TimeoutError:
        return 0, False, "Analysis timed out - please try with clearer images"
    except Exception as e:
        return 0, False, str(e)

# Similarity interpretation
def get_similarity_status(similarity):
    """Similarity interpretation"""
    if similarity > 80:
        return "Excellent Match - Same person", "success"
    elif similarity > 70:
        return "Very High Similarity - Likely same person", "success"
    elif similarity > 60:
        return "High Similarity - Probably same person", "warning"
    elif similarity > 50:
        return "Moderate Similarity - Could be same person", "warning"
    elif similarity > 40:
        return "Low-Moderate Similarity - Uncertain", "info"
    else:
        return "Low Similarity - Likely different people", "error"

# Results display
def display_results(similarity, img1, img2, processing_time=None):
    """Display results clearly"""
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        st.subheader("Reference Image")
        st.image(img1, use_container_width=True)
    
    with col2:
        st.subheader("Similarity Score")
        st.markdown(f"<h1 style='text-align:center; color: #1f77b4;'>{similarity:.1f}%</h1>", unsafe_allow_html=True)
        msg, level = get_similarity_status(similarity)
        getattr(st, level)(msg)
        
        if processing_time:
            st.info(f"Processed in {processing_time:.1f} seconds")
        
        st.progress(min(similarity/100, 1.0))
    
    with col3:
        st.subheader("Comparison Image")
        st.image(img2, use_container_width=True)

# Camera mode
def camera_mode():
    st.header("Upload Reference + Camera Photo")
    
    with st.expander("Tips for Best Results"):
        st.markdown("""
        - Use good lighting and clear faces
        - Front-facing photos work best
        - Avoid sunglasses or masks if possible
        - Analysis typically takes 3-8 seconds
        """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Reference Image")
        uploaded_file = st.file_uploader("Upload reference", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            ref_img = Image.open(uploaded_file).convert("RGB")
            st.image(ref_img, caption="Reference", use_container_width=True)
    
    with col2:
        st.subheader("Camera Photo")
        camera_photo = st.camera_input("Take photo")
        if camera_photo:
            camera_img = Image.open(camera_photo).convert("RGB")
            st.image(camera_img, caption="Camera", use_container_width=True)
    
    # Analysis
    if uploaded_file and camera_photo:
        st.markdown("---")
        if st.button("Analyze Similarity", use_container_width=True, type="primary"):
            start_time = time.time()
            
            with st.spinner("Analyzing faces..."):
                similarity, verified, error = calculate_similarity(ref_img, camera_img)
                
            processing_time = time.time() - start_time
            
            if error:
                st.error(f"Error: {error}")
            else:
                st.success("Analysis complete!")
                display_results(similarity, ref_img, camera_img, processing_time)

# Image comparison mode
def image_comparison_mode():
    st.header("Compare Two Images")
    
    col1, col2 = st.columns(2)
    with col1:
        img1_file = st.file_uploader("First Image", type=["jpg", "jpeg", "png"], key="img1")
        if img1_file:
            img1 = Image.open(img1_file).convert("RGB")
            st.image(img1, caption="Image 1", use_container_width=True)
    
    with col2:
        img2_file = st.file_uploader("Second Image", type=["jpg", "jpeg", "png"], key="img2")
        if img2_file:
            img2 = Image.open(img2_file).convert("RGB")
            st.image(img2, caption="Image 2", use_container_width=True)

    if img1_file and img2_file:
        st.markdown("---")
        if st.button("Compare Images", use_container_width=True, type="primary"):
            start_time = time.time()
            
            with st.spinner("Comparing faces..."):
                similarity, verified, error = calculate_similarity(img1, img2)
                
            processing_time = time.time() - start_time
            
            if error:
                st.error(f"Error: {error}")
            else:
                st.success("Comparison complete!")
                display_results(similarity, img1, img2, processing_time)

# About page
def about_page():
    st.header("About Face Similarity Detector")
    
    st.markdown("""
    This application uses advanced AI models to compare facial similarities between two images.
    
    ### Features
    - **Automatic face detection** and cropping for better accuracy
    - **Smart preprocessing** with contrast enhancement when needed
    - **Timeout protection** prevents long processing times
    - **Optimized processing** for best speed/accuracy balance
    
    ### Performance Optimization
    - **3-8 second analysis time** with balanced accuracy
    - **15-second timeout** limit to prevent hanging
    - **Efficient memory management** and cleanup
    
    ### Technical Details
    - **Model**: Facenet512 for balanced speed and accuracy
    - **Distance Metric**: Cosine similarity
    - **Image Processing**: OpenCV for face detection and enhancement
    - **Quality Settings**: Optimized JPEG compression for best results
    
    ### Accuracy Features
    - Face detection with automatic cropping
    - Contrast adjustment for low-light images
    - Minimum resolution requirements
    - Robust error handling and fallbacks
    
    ### Usage Tips
    - Use well-lit, front-facing photos for best results
    - Ensure faces are clearly visible and unobstructed
    - Higher quality images generally produce better accuracy
    - Remove glasses, hats, or face coverings when possible
    """)

# Main app
def main():
    st.title("Face Similarity Detector")
    st.markdown("**AI-powered facial recognition and comparison**")
    
    # Sidebar for mode selection
    st.sidebar.markdown("### Select Mode")
    mode = st.sidebar.radio(
        "Choose comparison method:",
        ["Camera + Upload", "Compare Two Images", "About"],
        index=0
    )
    
    if mode == "Camera + Upload":
        camera_mode()
    elif mode == "Compare Two Images":
        image_comparison_mode()
    else:
        about_page()

if __name__ == "__main__":
    main()