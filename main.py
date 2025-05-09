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
    st.title("Fantastic Fourier Transform")
    value = st.slider("Frequency", 0.0, 9.0, value=0.0, step=0.1)
    fig = cos_wave(value)
    st.plotly_chart(fig)

    st.plotly_chart(create_winding(value))

    st.plotly_chart(create_lena_fft())

    option = st.selectbox("Select a basic Shape:",
                           ('Star', 'Square', 'Circle', 'X'), index=None)

    st.plotly_chart(create_fft_showcase(option))

    st.plotly_chart(create_freq_seq())
    st.plotly_chart(create_orientation_seq())
    st.plotly_chart(create_amplitude_seq())
    st.plotly_chart(create_kspace())

    option2 = st.selectbox("Select a MRI image:",
                          ["Image"+str(i) for i in range(10)], index=None)
    number = st.number_input("Input a mask size:", value=0.0, step=0.1)

    o_image, fft, r_image = create_mri_reconstruction(75, number)

    col1, col2 = st.columns(2)

    col1.plotly_chart(fft)
    col2.plotly_chart(r_image)

    with st.expander("Click to reveal the original image:"):
        st.image(o_image)

if __name__ == "__main__":
    main()