import numpy as np
import cv2
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import time

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

def create_freq_img(freq, angle, H, W):

    x = np.linspace(-W/2, W/2, W) # X and Y pixel coordinates 
    y = np.linspace(-H/2, H/2, H)

    # Create a meshgrid
    X, Y = np.meshgrid(x, y)

    angle_rad = np.radians(angle)
    
    # Create the grating
    grating = np.sin(2 * np.pi * freq * (X * np.cos(angle_rad) + Y * np.sin(angle_rad)))
    return grating

def fft_freq_img(img):
    return np.abs(np.fft.fftshift(np.fft.fft2(img))) # compute fft
    
def create_freq_chart():

    fig = make_subplots(1, 2) # define number of subplots

    # sinusoid grating params
    freq, angle, H, W = 0, 0, 100, 100

    display_img = create_freq_img(freq, angle, H, W) #create sinusoidal grating
    fft_img = fft_freq_img(display_img) # compute fft image

    seq = [display_img, fft_img / np.max(fft_img)]

    fig = px.imshow(np.array(seq), color_continuous_scale='gray', facet_col=0)

    steps = []

    for f in range(0, 51):  # Add frequencies from 0 to 50 
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



    
