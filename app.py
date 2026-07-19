"""
Streamlit frontend for the Tic-Tac-Toe board reader.

Run with:
    streamlit run app.py
"""

import numpy as np
import cv2
import streamlit as st
from ttt_reader import process_image_from_array, print_results_as_lines, draw_results

st.set_page_config(page_title="Tic-Tac-Toe Reader", page_icon="⭕", layout="centered")

st.title("Tic-Tac-Toe Board Reader")
st.write("Upload an image of a 3x3 tic-tac-toe board and I'll tell you what's in every cell.")

uploaded_file = st.file_uploader("Upload board image", type=["png", "jpg", "jpeg", "bmp"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Uploaded photo", use_container_width=True)

    results, board, cell_h, cell_w = process_image_from_array(img)

    st.image(cv2.cvtColor(board, cv2.COLOR_BGR2RGB), caption="Detected board (cropped & straightened)",
              use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Text output")
        for line in print_results_as_lines(results):
            st.text(line)

    with col2:
        st.subheader("Drawn over image")
        annotated = draw_results(board.copy(), results, cell_h, cell_w, out_path=None)
        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)
else:
    st.info("Waiting for an image upload...")