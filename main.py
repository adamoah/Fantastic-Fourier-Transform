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
    st.write("The Fourier Transform has many applications for analyzing 2D signals, or images. " \
    "In the same way a 1D FFT converts a 1D spatial signal into a frequency signal, a 2D FFT converts "\
    "a spatial image into a frequency image. While images in the frequency domain can be intimidating at first "\
    "understanding how frequency components relate to spatial components play a key role in image processing "\
    "techniques such as noise reduction, compression, feature detection, and more.")

    st.plotly_chart(create_lena_fft())

    
    st.subheader("How to compute a 2D FFT")
    st.markdown('''Algorithmically, the 2D FFT is very much an extention of the 1D case and involves 4 key steps:''')
    st.markdown('''
                    1. Convert the image to grayscale (This converts your image dimensions from (H, W, C) to (H, W) ensuring it is actually 2D)
                    2. Compute the 1D FFT of each column of the grayscale image
                    3. Compute the 1D FFT of each row of the image from the previous step
                    4. To get amplitude information, we first shift the low frequencies to the center and take the absolute value of the real component of the FFT.''')
    st.text("The heatmaps below visualize each of these steps to derive the FFT amplitudes for a variety of basic shapes. " \
            "Use the drop down menu below to select different shapes and observe how their FFTs are derived")

    tab1, tab2, tab3, tab4 = st.tabs(["Basic Shapes", "Grating Changes", "MRI", "Audio"])
    data = np.random.randn(10, 1)
    tab1.subheader("test")
    tab1.line_chart(data)

    option = st.selectbox("Select a basic Shape:",
                        ('Star', 'Square', 'Circle', 'X'), index=None)

    st.plotly_chart(create_fft_showcase(option))

    st.text("In the same way that the FFT shows us a 1D signal can be decomposed into a sum of sine and cosine wave, a " \
            "2D signal can be decomposed into a sum of 2D sine waves, sometimes referred to as a sinusoidal grating.")
    st.text("The following visual shows how changing different aspects of the grating affect what you see in the frequency space. " \
            "You can use the slider to view the images for each value or use the play button to step through all values.")

    st.plotly_chart(create_freq_seq())
    st.text("Changing the frequency of the grating has a proportional effect on the magnitude of the high frequencies and an inverse "
    "effect on the low freqencies. In the spatial image, increased frequency signifies rapid changes in pixel intensity (brightness).")
    st.plotly_chart(create_orientation_seq())
    st.text("Changing the orientation of the grating has a proportional effect on the angle of the magnitudes in the frequency space. " \
            "In the spatial image, the orientation signifies the direction of the observed pixel intensity change.")
    st.plotly_chart(create_amplitude_seq())
    st.text("Changing the magnitude of the grating changes the brightness of the image and scales the magnitudes" \
            "of the FFT by the same factor. This is paticularly relevant to the DC component of the FFT (located in the center), which " \
            "represents the \"average brightness\" of the image.")

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
