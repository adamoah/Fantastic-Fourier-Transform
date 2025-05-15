import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import cv2 as cv
import matplotlib.pyplot as plt
from functions.test import *
import os

def main(): 
    '''
    main function displays the webside
    '''
    st.title("The Fantastic Four(ier) Transform")
    st.subheader("And its Applications")

    value = st.slider("Frequency", 0.0, 3.0, value=0.1, step=0.1)
    fig, x, cos_wave = create_cos_wave(value)
    st.plotly_chart(fig)

    st.plotly_chart(create_winding(x, cos_wave))


    st.header("Fourier Transform in 2D")
    st.write("The Fourier Transform has many applications for analyzing 2D signals, or images." \
    "In the same way a 1D FFT converts a 1D spatial signal into a frequency signal, a 2D FFT converts"\
    "a spatial image into a frequency image. While images in the frequency domain can be intimidating at first"\
    "understanding how frequency components relate to spatial components play a key role in image processing"\
    "techniques such as noise reduction, compression, feature detection, and more.")

    st.plotly_chart(create_lena_fft())

    option = st.selectbox("Select a basic Shape:",
                           ('Star', 'Square', 'Circle', 'X'), index=None)
    
    st.subheader("How to compute a 2D FFT")
    st.text("Algorithmically, the 2D FFT is very much an extention of the 1D case and involves 4 key steps:")
    st.html("<ul>"\
            "<pre><li><strong>Step 1:</strong> Convert the image to grayscale (This converts your image dimensions from (H, W, C) to (H, W)<br>ensuring it is actually 2D)<br></li></pre>"\
            "<pre><li><strong>Step 2:</strong> Compute the 1D FFT of each column of the grayscale image<br></li></pre>"\
            "<pre><li><strong>Step 3:</strong> Compute the 1D FFT of each row of the image from the previous step<br></li></pre>"\
            "<pre><li><strong>Step 4:</strong> To get amplitude information, we first shift the low frequencies to the center and take the\n\tabsolute value of the real component of the FFT.</li></ul></pre>")

    st.plotly_chart(create_fft_showcase(option))

    st.plotly_chart(create_freq_seq())
    st.plotly_chart(create_orientation_seq())
    st.plotly_chart(create_amplitude_seq())
    
    html = get_kspace_html()
    st.components.v1.html(html, height=500)
    

    tumor_files = sorted(os.listdir("./data/Tumors/"))
    tumor_choices = ['Tumor 100', 'Tumor 120', 'Tumor 22', 'Tumor 243', 'Tumor 36', 'Tumor 65', 'Tumor 7', 'Tumor 75', 'Tumor 89', 'Tumor 97']
    tumor_ffts = create_mri_ffts()


    tumor = st.selectbox("Select a MRI image:",
                          tumor_choices, index=None)
    number = st.number_input("Input a mask size:", value=0.0, step=0.1)


    if tumor != None:
        idx = tumor_choices.index(tumor)
        fft, r_image = create_mri_reconstruction(tumor_ffts[idx], number)

        col1, col2 = st.columns(2)

        col1.plotly_chart(fft)
        col2.plotly_chart(r_image)

        with st.expander("Click to reveal the original image:"):
            st.image("./data/Tumors/"+tumor_files[idx])

if __name__ == "__main__":
    main()