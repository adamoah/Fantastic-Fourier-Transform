# Fantastic-Fourier-Transform

Visuals inspired by:  
https://www.youtube.com/watch?v=spUNpyF58BY
  
https://github.com/thatSaneKid/fourier/blob/master/Fourier%20Transform%20-%20A%20Visual%20Introduction.ipynb

Piano audio clips from: 
https://github.com/fuhton/piano-mp3

Kspace Data from:
https://fastmri.med.nyu.edu/

MRI Images from:
https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection/data

This is a collaborative final project by Andrews Damoah and Richard Sembria for CMSC471 at the
University of Maryland.

Idea proposed was to create a mini-lesson for understanding the mechanics of the Fourier transform
and its applications in an intuitive way, without the mathematics. No need for an intensive math
background.

Work Distribution:
  
Andrews Damoah: 
Created 3 sections (2D FFT, Sinusoidal Grating, MRI Example) . 1) A introduction to the 2D Fourier transform and a basic example using shapes encoded in matrices. 
2) A section on sinusoidal grating showcasing the effects of changing certain parameters on the 2D frequency domain, namely the frequency, orientation, and magnitude. 
Visualizations are interactable and real-time. 3) An MRI section with a small introduction, example, a interactive "game" involving 10 MRI scans of different parts of the 
body with tumors. Optimizations for webpage loading and "piano game" data processing. Added feature for being able to play a reconstructed audio track of the 
chosen combination of notes in the "piano game".

Richard Sembria:
Created 2 sections and formatted the webpage (Visualizing 1D FFT, Audio Example). 1) An introduction and interactive visualization for the basic principles of the
Fourier transform. Users can change the frequency of the original function (a cosine wave) and the winding graph to see the effects. 2) A example application of the 
Fourier transform involving audio. Users can listen to the provide audio track (C major chord) and observe the FFT of the given audio track. Included is a game that
users can play to try to reconstruct the chord with the provided notes (in C major scale). Webpage was condensed into tabs to avoid length and consequently, easier 
and more organized access to different sections of the project.

Website can be found here: https://fantastic-fourier-transform-vpkqfpqxqcappzbfqj9uznb.streamlit.app/

To run it locally make sure you have python and/or a IDE of your choice.
First download the required packages in your terminal
```
pip install -r ./requriements.txt
```
You can then launch a local streamlit app by running the terminal command
```
python -m streamlit run ./main.py
```

