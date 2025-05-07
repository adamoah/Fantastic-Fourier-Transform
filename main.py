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
                           ('Star', 'Square', 'Circle', 'X'))

    st.plotly_chart(create_fft_showcase(option))

    st.plotly_chart(create_freq_seq())
    st.plotly_chart(create_orientation_seq())
    st.plotly_chart(create_amplitude_seq())
    st.plotly_chart(create_kspace())


if __name__ == "__main__":
    main()