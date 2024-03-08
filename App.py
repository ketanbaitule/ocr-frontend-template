import streamlit as st
import numpy as np
import cv2
from PIL import Image
import pytesseract
import easyocr

def preprocess_img(image):
    # gray=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # return gray
    return image

@st.cache_resource
def get_easyocr_model():
    reader = easyocr.Reader(['ch_sim','en'])
    return reader

st.title("OCR MODEL")

uploaded_files = st.file_uploader("Upload Image File", accept_multiple_files=True, type=["png", "jpg", "jpeg"])

with st.sidebar:
    st.subheader("Configuration")
    model_selection = st.radio(
        "Choose Model: ", 
        ("Tesseract", "EasyOCR")
    )

if uploaded_files is not None:
    st.divider()
    st.subheader(f"Model: {model_selection}")
    for index, uploaded_file in enumerate(uploaded_files):
        img_file = Image.open(uploaded_file)
        st.subheader(f"Image {index + 1}: {uploaded_file.name}")
        col1, col2 = st.columns(2)
        with col1:
            st.image(uploaded_file, caption=uploaded_file.name)
        with col2:
            text = ""
            if model_selection == "Tesseract":
                gray = preprocess_img(img_file)
                text = pytesseract.image_to_string(gray)
            elif model_selection == "EasyOCR":
                reader = get_easyocr_model()
                data = reader.readtext(np.array(img_file))
                with st.expander("Get Bounding table", expanded=False):
                    st.table(data)
                with st.expander("Image", expanded=False):
                    # For Colors
                    color = st.color_picker('Pick A Color', '#00f900')
                    r, g, b = bytes.fromhex(color[1:])
                    # For Thickness
                    thickness = st.number_input("Select Thickness", step=1, min_value=1, value=5)
                    img_file_bounding_box = np.array(img_file.copy())
                    for bb in (item[0] for item in data):
                        (tl, tr, br, bl) = bb
                        tl = (int(tl[0]), int(tl[1]))
                        tr = (int(tr[0]), int(tr[1]))
                        br = (int(br[0]), int(br[1]))
                        bl = (int(bl[0]), int(bl[1]))
                        cv2.rectangle(img_file_bounding_box, tl, br, color=(r, g, b), thickness=int(thickness))
                    st.image(img_file_bounding_box, caption=f"{uploaded_file.name} - Bounding Box")
                text = "\n".join(item[1] for item in data)
            st.code(text)
        st.divider()