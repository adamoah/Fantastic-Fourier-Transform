import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from streamlit_image_comparison import image_comparison
import cv2 as cv
import matplotlib.pyplot as plt
from functions.test import *
import os

def main(): 
    '''
    main function displays the webside
    '''
    st.html(
    """<style>
        h1 {
            font-size: 40px;
            text-align: center;}
        h2 {
            font-size: 25px;
            text-align: center;}
        </style>
        
        <h1>The Fantastic Four(ier) Transform</h1>
        <h2>And Its Applications</h2>

        <p>The Fourier transform is a wonderful mathematical tool developed by Jean-Baptiste Joseph Fourier in the early 1800s whilst 
        studying heat transfer. This tool turns functions of time into functions of frequency. But exactly what does this mean? 
        Fortunately, we will not be diving into a detailed mathematical explanation for your understanding of the Fourier transform.
        Instead, we will be using you eyes, hands, and ears!</p>"""
    )
    
    st.text("Use the slider to reveal the image!")
    image_comparison(img1="./data/fantasticfft.png", img2="./data/fantastic4.jpeg", starting_position=99.9, show_labels=False, make_responsive=False)
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Visualizing 1D FFT", "2D FFT", "Sinusoidal Grating", "Audio Example", "MRI Example"])
    
    with tab1:
        st.html(
        """
        <p>To explain the Fourier transform visually, let us begin with a very simple case. Let us take the cosine wave seen below. A slider
        is included to modify the frequency of our function. Assuming some math background, you can see that the wavelength is shortened 
        over our time interval. If you do not have a math background, just think of our function as a line going up and down more times 
        during an amount of time. This will be our function of time.</p>.
        """
        )
        value = st.slider("Frequency", 0.0, 3.0, value=1.0, step=0.1)
        fig, x, cos_wave = create_cos_wave(value)
        st.plotly_chart(fig)

        st.html(
        """
        <p>Now how can we turn this into a function of frequency? The inuition behind the Fourier transform is to wrap our function of
        time (shown above) around the origin seen below on the left. This particular resulting graph holds a special property however: 
        we can choose how often we wrap our original function around! We will denote this our "winding frequency". We now have two 
        distinct values in our analysis of the Fourier transform, the original frequncy and our winding frequency. Now what happens if 
        we take the center of mass (sum of all x-positions of our points divided by the sum of all y-positions of our points) of our new 
        graph? Obeserve below on the right.</p>
        """
        )
        
        st.plotly_chart(create_winding(x, cos_wave))

        st.html(
        """
        <p>As our winding frequency varies, so does our center of mass! Can you notice something in particular? The x-position of our 
        center of mass holds a special relationship with the winding frequency of our graph on the left and most important the frequency
        of our original graph. This is our function of frequency! Let us see if you can finalize the connection between the three.</p>
        """
        )

    with tab2: # tab with high level explaination on how the 2D FFT work
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
    
    with tab3: # tab with the 2d sine/cosine wave visuals
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
    
    with tab4:
        st.text("Sounds can also be turned into data as well. T")
        
        st.audio("data/pianoWav/c-major-chord.wav", format="audio/mpeg", loop=False)
        
        st.text("Below, you can find the discrete fast Fourier transform of the audio clip. Can you notice something?")

        piano, sr = audio_to_data("c-major-chord.wav")
        st.plotly_chart(audio_graph(piano, sr, 'svg', 0.025))
        
        st.text("There seems to be peaks at certain frequencies of the audio clip. How about we take a look at the frequencies of the notes in the C major scale?" \
               " We've included an audio clip in case you wanted to figure this out using your ears.")

        audio = ["C4.wav",
                 "D4.wav",
                 "E4.wav",
                 "F4.wav",
                 "G4.wav",
                 "A4.wav",
                 "B4.wav",
                 "C5.wav"]
        
        
        notes = []
        graph_data = [];
        
        
        for clip in audio:
            notes.append(audio_to_data(clip))

        for note in notes:
            graph_data.append(audio_fft(note[0], note[1], 0.025))

            
        length = len(graph_data[0][0])
        
        curr_graph = graph_data[0][0], graph_data[0][1], [0] * length


        col1, col2, col3, col4, col5, col6, col7, col8 = st.columns(8)

        with col1:
            C4 = st.toggle("shart1")
        with col2:
            D4 = st.toggle("shart2")
        with col3:
            E4 = st.toggle("shart3")
        with col4:
            F4 = st.toggle("shart4")
        with col5:
             G4 = st.toggle("shart5")
        with col6:
            A4 = st.toggle("shart6")
        with col7:
            B4 = st.toggle("shart7")
        with col8:
            C5 = st.toggle("shart8")

        flags = [False, False, False, False, False, False, False, False]
        
        if C4:
            flags[0] = True
            for i in range(length):
                curr_graph[2][i] += graph_data[0][2][i]
        else:
            if flags[0]:
                for i in range(length):
                    curr_graph[2][i] -= graph_data[0][2][i]
                    
        if D4:
            flags[1] = True
            for i in range(length):
                    curr_graph[2][i] += graph_data[1][2][i]
        else:
            if flags[1]:
                for i in range(length):
                    curr_graph[2][i] -= graph_data[1][2][i]
                    
        if E4:
            flags[2] = True
            for i in range(length):
                curr_graph[2][i] += graph_data[2][2][i]
        else:
            if flags[2]:
                for i in range(length):
                    curr_graph[2][i] -= graph_data[2][2][i]
                    
        if F4:
            flags[3] = True
            for i in range(length):
                curr_graph[2][i] += graph_data[3][2][i]
        else:
            if flags[3]:
                for i in range(length):
                    curr_graph[2][i] -= graph_data[3][2][i]
                    
        if G4:
            flags[4] = True
            for i in range(length):
                curr_graph[2][i] += graph_data[4][2][i]
        else:
            if flags[4]:
                for i in range(length):
                    curr_graph[2][i] -= graph_data[4][2][i]
                    
        if A4:
            flags[5] = True
            for i in range(length):
                curr_graph[2][i] += graph_data[5][2][i]
        else:
            if flags[5]:
                for i in range(length):
                    curr_graph[2][i] -= graph_data[5][2][i]
                    
        if B4:
            flags[6] = True
            for i in range(length):
                curr_graph[2][i] += graph_data[6][2][i]
        else:
            if flags[6]:
                for i in range(length):
                    curr_graph[2][i] -= graph_data[6][2][i]
                        
        if C5:
            flags[7] = True
            for i in range(length):
                curr_graph[2][i] += graph_data[7][2][i]
        else:
            if flags[7]:
                for i in range(length):
                    curr_graph[2][i] -= graph_data[7][2][i]

        with st.container():
        
            # You can call any Streamlit command, including custom components:
            df = pd.DataFrame({'x': curr_graph[0][:curr_graph[1]], 'y': curr_graph[2][:curr_graph[1]]})
            fig = px.line(df, x="x", y="y")
            st.plotly_chart(fig)

    
    with tab5:
        st.subheader("2D Application: MRI")

        st.text("One application of FFT is for the reconstruction of Magnetic Resonance Imaging (MRI) images. " \
        "When an MR scan is conducted, magnetic fields and radio waves are used to spin protons in the" \
        "body out of equilibrium and record the emitted waves as the protons realign. By using a controlled gradient to slightly " \
        "modify the magnetic field of the machine, it records waves from protons at different frequencies (Sound familiar?). " \
        "By doing this across the X, Y, and Z axes, you create whats known as a k-space matrix, which stores amplitude and phase " \
        "information of the recorded spatial frequiences. This is essential the same information output by the Fourier Transform " \
        "meaning if we can use FFT to derive a frequency image from a spatail one, doing the inverse of the FFT means " \
        "we can derive a spatial one from a frequency one.")
        st.text("The visual below shows you the reconstructed spatial image from k-space data after an inverse FFT " \
        "for various 2D cross-sections of a scanned knee. You can see how as the k-space gets more \"filled out\" " \
        "we see more details in the reconstructed image.")

        html = get_kspace_html()
        st.components.v1.html(html, height=550)
        
        st.text("Another interesting application of the FFT is filtering. By removing or masking certain regions of the " \
        "frequency image, we can preserve certain aspects of the spatial image. More specifically, by masking the low " \
        "freqiency components (which encode the color contrast and \"smoothness\" of the image) we can preseve just the edge details and textures"
        "(This is also know as a high-pass filter). " \
        "In MRI, this can be useful for detecting abnormalities such as tumors, inflammation, or structural deformities.")
        st.text("The visual below contains a set of 10 MRI brain scan of patients with tumors. For each image you can specify the mask size, " \
        "and view the masked FFT and reconstructed image. Using the mask, your goal is to try to find the tumor in the reconstructed image. " \
        "If you think you have found it, click on the reveal tab to see the original image.")

        tumor_files = sorted(os.listdir("./data/Tumors/"))
        tumor_choices = ['Tumor 100', 'Tumor 120', 'Tumor 22', 'Tumor 243', 'Tumor 36', 'Tumor 65', 'Tumor 7', 'Tumor 75', 'Tumor 89', 'Tumor 97']
        tumor_ffts = create_mri_ffts()
    
    
        tumor = st.selectbox("Select a MRI image:",
                              tumor_choices, index=None)
        number = st.number_input("Input a mask size:", value=50, step=1)
    
    
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
