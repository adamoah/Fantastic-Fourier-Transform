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

    tab1, tab2, tab3, tab4 = st.tabs(["2D FFT", "Sinusoidal Grating", "Audio Example", "MRI Example"])
    
    with tab1: # tab with high level explaination on how the 2D FFT work
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
                "Use the drop down menu below to select different patterns and observe how their FFTs are derived")

        option = st.selectbox("Select a basic pattern:",
                            ('Vertical Stripe', 'Polka Dots', 'Plus Sign', 'X'), index=None)
        if option != None:
            st.plotly_chart(create_fft_showcase(option))
            st.text("Notice how the different patterns affect which regions have higher amplitude. " \
            "We will explore why this is in the next section")
    
    with tab2: # tab with the 2d sine/cosine wave visuals
        st.subheader("2D Sine and Cosine Waves")
        st.text("In the same way that the FFT shows us a 1D signal can be decomposed into a sum of sine and cosine wave, a " \
                "2D signal can be decomposed into a sum of 2D sine waves, sometimes referred to as a sinusoidal grating.")
        st.text("The following visual shows how changing different aspects of the grating affect what you see in the frequency space.")
    
        st.plotly_chart(create_freq_seq())
        st.text("Changing the frequency of the grating has a proportional effect on the magnitude of the high frequencies and an inverse "
        "effect on the low freqencies. In the spatial image, increased frequency signifies rapid changes in pixel intensity (brightness).")
        st.plotly_chart(create_orientation_seq())
        st.text("Changing the orientation of the grating has a proportional effect on the angle of the magnitudes in the frequency space. " \
                "In the spatial image, the orientation signifies the direction of the observed pixel intensity change.")
        st.plotly_chart(create_amplitude_seq())
        st.text("Changing the magnitude of the grating changes the brightness of the image and scales the magnitudes " \
                "of the FFT by the same factor. This is paticularly relevant to the DC component of the FFT (located in the center), which " \
                "represents the \"average brightness\" of the image.")
    

    with tab3:
        st.text("Sounds can also be turned into data as well. T")
        
        st.audio("data/pianoWav/c-major-chord.wav", format="audio/mpeg", loop=False)
        
        st.text("Below, you can find the discrete fast Fourier transform of the audio clip. Can you notice something?")
        
        st.plotly_chart(audio_showcase("c-major-chord.wav"))
        
        st.text("There seems to be peaks at certain frequencies of the audio clip. How about we take a look at the frequencies of the notes in the C major scale?" \
               " We've included an audio clip in case you wanted to figure this out using your ears.")
        
        stab1, stab2, stab3, stab4, stab5, stab6, stab7, stab8 = st.tabs(["C4", "D4", "E4", "F4", "G4", "A4", "B4", "C5"])
        with stab1:
            st.audio("data/pianoWav/C4.wav", format="audio/mpeg", loop=False)
            st.plotly_chart(audio_showcase("C4.wav"))
        with stab2:
            st.audio("data/pianoWav/D4.wav", format="audio/mpeg", loop=False)
            st.plotly_chart(audio_showcase("D4.wav"))
        with stab3:
            st.audio("data/pianoWav/E4.wav", format="audio/mpeg", loop=False)
            st.plotly_chart(audio_showcase("E4.wav"))
        with stab4:
            st.audio("data/pianoWav/F4.wav", format="audio/mpeg", loop=False)
            st.plotly_chart(audio_showcase("F4.wav"))
        with stab5:
            st.audio("data/pianoWav/G4.wav", format="audio/mpeg", loop=False)
            st.plotly_chart(audio_showcase("G4.wav"))
        with stab6:
            st.audio("data/pianoWav/A4.wav", format="audio/mpeg", loop=False)
            st.plotly_chart(audio_showcase("A4.wav"))
        with stab7:
            st.audio("data/pianoWav/B4.wav", format="audio/mpeg", loop=False)
            st.plotly_chart(audio_showcase("B4.wav"))
        with stab8:
            st.audio("data/pianoWav/C5.wav", format="audio/mpeg", loop=False)
            st.plotly_chart(audio_showcase("C5.wav"))

    with tab4:
        st.subheader("2D Application: MRI")
        html = get_kspace_html()
        st.components.v1.html(html, height=550)
        
    
        tumor_files = sorted(os.listdir("./data/Tumors/"))
        tumor_choices = ['Tumor 100', 'Tumor 120', 'Tumor 22', 'Tumor 243', 'Tumor 36', 'Tumor 65', 'Tumor 7', 'Tumor 75', 'Tumor 89', 'Tumor 97']
        tumor_ffts = create_mri_ffts()
    
    
        tumor = st.selectbox("Select a MRI image:",
                              tumor_choices, index=None)
        number = st.number_input("Input a mask size:", value=0, step=1)
    
    
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
