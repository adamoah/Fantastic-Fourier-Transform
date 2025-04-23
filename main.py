import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import cv2 as cv
import matplotlib.pyplot as plt
from functions.test import *

def main():
    '''
    main function displays the webside
    '''
    st.title("Hello World")
    # value = st.slider("Brightness", 0, 10, value=1, step=1)
    fig = load()
    st.plotly_chart(fig)
    value = st.slider("Frequency", 0, 10, value=0, step=1)
    '''
    image = create_freq_img(value)
    fft_img = fft_freq_img(image).astype(np.uint8)
    norm_fft = cv.applyColorMap(fft_img, cv.COLORMAP_VIRIDIS)

    col1, _, col2 = st.columns(3)
    '''
    st.plotly_chart(create_freq_chart())


if __name__ == "__main__":
    main()