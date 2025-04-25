import numpy as np
import cv2
import streamlit as st

import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

import pandas as pd
import time
import json

@st.cache_data
def load():

    # x and y data
    x = np.arange(0, 20, 0.1)
    y = np.cos(x)
    
    df = pd.DataFrame({'x': x, 'y': y}) # create dataframe

    fig = px.line(df, x='x', y='y') # basic visualization

    steps = []
    for f in range(0, 51):  # Add frequencies from 0 to 5 (in 0.1 intervals)
        step = {
            "method": "update",
            "args": [{"y": [np.sin(0.1 * f * x)]}],
            "label": str(f*0.1),
        }
        steps.append(step)

    # Create the slider
    sliders = [
        {
            "active": 0,
            "pad": {"t": 50},
            "steps": steps,
            "currentvalue": {"prefix": "Frequency: "},
        }
    ]

    # Update the layout with the slider
    fig.update_layout(sliders=sliders)

    return fig # retrun vis


def create_freq_img(freq, angle, mag, H, W):

    freq = 100 / freq
    angle = angle + 90

    x = np.arange(W) # X and Y pixel coordinates 
    y = np.arange(H)

    # Create a meshgrid
    X, Y = np.meshgrid(x, y)

    # Get the appropriate gradient
    gradient = np.sin(angle * np.pi / 180) * X - np.cos(angle * np.pi / 180) * Y    
    
    # Create the grating
    grating = mag * np.sin((2 * np.pi * gradient) / freq + (0 * np.pi) / 180)
    return grating

@st.cache_data
def fft_freq_img(img):
    return np.abs(np.fft.fftshift(np.fft.fft2(img))) # compute fft

@st.cache_data
def create_freq_chart():

    fig = make_subplots(1, 2) # define number of subplots

    # sinusoid grating params
    freq, angle, H, W = 1, 0, 100, 100

    display_img = create_freq_img(freq, angle, H, W) #create sinusoidal grating
    fft_img = fft_freq_img(display_img) # compute fft image

    seq = [display_img, fft_img / np.max(fft_img)]

    fig = px.imshow(np.array(seq), color_continuous_scale='gray', facet_col=0)

    fig.layout.annotations[0]['text'] = "Spatial Domain"
    fig.layout.annotations[1]['text'] = "Frequency Domain"


    steps, steps2 = [], []

    for f in range(1, 11):  # Add frequencies from 0 to 10 
        display_img = create_freq_img(f, angle, H, W) 
        fft_img = fft_freq_img(display_img)
        seq = [display_img, fft_img / np.max(fft_img)]
        step = {
            "method": "update",
            "args": [{"z": seq}],
            "label": str(f),
        }
        steps.append(step)

    # Create the slider
    sliders = [
        {
            "active": 0,
            "pad": {"t": 50},
            "steps": steps,
            "currentvalue": {"prefix": "Frequency: "},
        }
    ]

    fig.update_layout(sliders=sliders, coloraxis_showscale=False) # update figure
    fig.update_xaxes(showticklabels=False) # remove tick labels
    fig.update_yaxes(showticklabels=False)

    return fig



def create_freq_seq(angle=0, mag=1, H=100, W=100, freq=None):
    '''
    Creates a sequence of sinusoid grating and corresponding FFT with variable frequencies
    '''

    frames = np.empty(shape=(5, 2, H, W))
    # Create animation frames
    for i, f in enumerate(range(0, 21, 5)):

        freq = 1 if f == 0 else f

        display_img = create_freq_img(freq, angle, mag, H, W) #create sinusoidal grating
        fft_img = fft_freq_img(display_img) # compute fft image

        # append images to frame
        frames[i, 0, :, :] = display_img
        frames[i, 1, :, :] = fft_img / np.max(fft_img) 
    
    fig = px.imshow(np.array(frames), color_continuous_scale='gray', animation_frame=0, facet_col=1, height=500) # create animation
    
    # plot text labels
    fig.layout.annotations[0]['text'] = "Spatial Domain"
    fig.layout.annotations[1]['text'] = "Frequency Domain"

    # Open and read the JSON file
    with open("./Fantastic-Fourier-Transform/data/slider_jsons/freq.JSON", 'r') as file:
        sliders = json.load(file)

    fig.update_layout(sliders=[sliders], coloraxis_showscale=False) # update figure layout

    # update figure axes
    fig.update_xaxes(title_text='X Pixel', title_standoff=5, showticklabels=False, row=1, col=1)
    fig.update_xaxes(title_text='X Frequency', title_standoff=5, showticklabels=False, row=1, col=2)
    fig.update_yaxes(title_text='Y Pixel', title_standoff=5, showticklabels=False, row=1, col=1)
    fig.update_yaxes(title_text='Y Frequency', title_standoff=5, showticklabels=False, row=1, col=2)

    # pio.write_html(fig, file="./Fantastic-Fourier-Transform/data/frequency.html", auto_play=True)

    # Build the figure
    return fig


def create_orientation_seq(freq=7, mag=1, H=100, W=100, angle=None):
    '''
    Creates a sequence of sinusoid grating and corresponding FFT with variable orientations
    '''

    frames = np.empty(shape=(8, 2, H, W))
    # Create animation frames
    for i, angle in enumerate(range(0, 360, 45)):

        display_img = create_freq_img(freq, angle, mag, H, W) #create sinusoidal grating
        fft_img = fft_freq_img(display_img) # compute fft image

        # append images to frame
        frames[i, 0, :, :] = display_img
        frames[i, 1, :, :] = fft_img / np.max(fft_img) 
    
    fig = px.imshow(np.array(frames), color_continuous_scale='gray', animation_frame=0, facet_col=1, height=500) # create animation
    fig.layout.annotations[0]['text'] = "Spatial Domain"
    fig.layout.annotations[1]['text'] = "Frequency Domain"
    
    # Open and read the JSON file
    with open("./Fantastic-Fourier-Transform/data/slider_jsons/angle.JSON", 'r') as file:
        sliders = json.load(file)

    fig.update_layout(sliders=[sliders], coloraxis_showscale=False) # update figure    
    
    # update figure axes
    fig.update_xaxes(title_text='X Pixel', title_standoff=5, showticklabels=False, row=1, col=1)
    fig.update_xaxes(title_text='X Frequency', title_standoff=5, showticklabels=False, row=1, col=2)
    fig.update_yaxes(title_text='Y Pixel', title_standoff=5, showticklabels=False, row=1, col=1)
    fig.update_yaxes(title_text='Y Frequency', title_standoff=5, showticklabels=False, row=1, col=2)

    # pio.write_html(fig, file="./Fantastic-Fourier-Transform/data/orientation.html", auto_play=True)

    # Build the figure
    return fig


def create_amplitude_seq(freq=7, angle=45, H=100, W=100, mag=None):
    '''
    Creates a sequence of sinusoid grating and corresponding FFT with variable frequencies
    '''

    frames = np.empty(shape=(5, 2, H, W))
    # Create animation frames
    for i, mag in enumerate(range(1, 6)):
    
        display_img = create_freq_img(freq, angle, mag*0.2, 100, 100) #create sinusoidal grating
        fft_img = fft_freq_img(display_img) # compute fft image

        # append images to frame
        frames[i, 0, :, :] = display_img
        frames[i, 1, :, :] = fft_img / np.max(fft_img) 
    
    fig = px.imshow(np.array(frames), color_continuous_scale='gray', animation_frame=0, facet_col=1, height=500) # create animation
    fig.layout.annotations[0]['text'] = "Spatial Domain"
    fig.layout.annotations[1]['text'] = "Frequency Domain"

    # Open and read the JSON file
    with open("./Fantastic-Fourier-Transform/data/slider_jsons/mag.JSON", 'r') as file:
        sliders = json.load(file)

    fig.update_layout(sliders=[sliders], coloraxis_showscale=False) # update figure
    
    # update figure axes
    fig.update_xaxes(title_text='X Pixel', title_standoff=5, showticklabels=False, row=1, col=1)
    fig.update_xaxes(title_text='X Frequency', title_standoff=5, showticklabels=False, row=1, col=2)
    fig.update_yaxes(title_text='Y Pixel', title_standoff=5, showticklabels=False, row=1, col=1)
    fig.update_yaxes(title_text='Y Frequency', title_standoff=5, showticklabels=False, row=1, col=2)

    # pio.write_html(fig, file="./Fantastic-Fourier-Transform/data/amplitude.html", auto_play=True)

    # Build the figure
    return fig