import numpy as np
import cv2
import streamlit as st

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import pandas as pd
import time
import json

@st.cache_data
def cos_wave(f):

    # x and y data
    x = np.arange(0,2,0.01) 
    y = np.cos(x * 2 * np.pi * f)
    
    df = pd.DataFrame({'x': x, 'y': y}) # create dataframe

    fig = px.line(df, x='x', y='y') # basic visualization

    '''
    steps = []
    for f in range(0, 6):  # Add frequencies from 0 to 5
        step = {
            "method": "update",
            "args": [{"y": [np.cos(2 * np.pi * f * x)]}],
            "label": str(np.round(f)),
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
    '''

    return fig # retrun vis

def create_winding(freq):

    fig = make_subplots(1, 2)

    sf_list = np.arange(0, 10, 0.1)
    x = np.arange(0,2,0.01) 
    cos_wave = np.cos(x * 2 * np.pi * freq)
    steps = []

    # compute winding x and y coordinates
    x_coords = [cos_wave[i]*np.cos(x[i]*5*2*np.pi) for i in range(len(x))]
    y_coords = [cos_wave[i]*np.sin(x[i]*5*2*np.pi) for i in range(len(x))]


    df = pd.DataFrame({'x': x_coords, 'y': y_coords}) # create dataframe

    fig = px.line(df, x='x', y='y', width=250, height=500) # basic visualization
    '''
    for idx, sf in enumerate(sf_list):


        step ={
            "method": "update",
            "args": [{"y": [np.cos(2 * np.pi * freq * x)]}],
            "label": str(np.round(freq)),
        }
    '''
    return fig

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


@st.cache_data
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
    
    fig = px.imshow(frames, color_continuous_scale='gray', animation_frame=0, facet_col=1, height=500) # create animation
    
    # plot text labels
    fig.layout.annotations[0]['text'] = "Spatial Domain"
    fig.layout.annotations[1]['text'] = "Frequency Domain"

    # Open and read the JSON file
    with open("./data/slider_jsons/freq.JSON", 'r') as file:
        sliders = json.load(file)

    fig.update_layout(sliders=[sliders], coloraxis_showscale=False) # update figure layout

    # update figure axes
    fig.update_xaxes(title_text='X Pixel', title_standoff=5, showticklabels=False, row=1, col=1)
    fig.update_xaxes(title_text='X Frequency', title_standoff=5, showticklabels=False, row=1, col=2)
    fig.update_yaxes(title_text='Y Pixel', title_standoff=5, showticklabels=False, row=1, col=1)
    fig.update_yaxes(title_text='Y Frequency', title_standoff=5, showticklabels=False, row=1, col=2)

    # pio.write_html(fig, file="./data/frequency.html", auto_play=True)

    # Build the figure
    return fig


@st.cache_data
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
    
    fig = px.imshow(frames, color_continuous_scale='gray', animation_frame=0, facet_col=1, height=500) # create animation
    fig.layout.annotations[0]['text'] = "Spatial Domain"
    fig.layout.annotations[1]['text'] = "Frequency Domain"
    
    # Open and read the JSON file
    with open("./data/slider_jsons/angle.JSON", 'r') as file:
        sliders = json.load(file)

    fig.update_layout(sliders=[sliders], coloraxis_showscale=False) # update figure    
    
    # update figure axes
    fig.update_xaxes(title_text='X Pixel', title_standoff=5, showticklabels=False, row=1, col=1)
    fig.update_xaxes(title_text='X Frequency', title_standoff=5, showticklabels=False, row=1, col=2)
    fig.update_yaxes(title_text='Y Pixel', title_standoff=5, showticklabels=False, row=1, col=1)
    fig.update_yaxes(title_text='Y Frequency', title_standoff=5, showticklabels=False, row=1, col=2)

    # pio.write_html(fig, file="./data/orientation.html", auto_play=True)

    # Build the figure
    return fig


@st.cache_data
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
    
    fig = px.imshow(frames, color_continuous_scale='gray', animation_frame=0, facet_col=1, height=500) # create animation
    fig.layout.annotations[0]['text'] = "Spatial Domain"
    fig.layout.annotations[1]['text'] = "Frequency Domain"

    # Open and read the JSON file
    with open("./data/slider_jsons/mag.JSON", 'r') as file:
        sliders = json.load(file)

    fig.update_layout(sliders=[sliders], coloraxis_showscale=False) # update figure
    
    # update figure axes
    fig.update_xaxes(title_text='X Pixel', title_standoff=5, showticklabels=False, row=1, col=1)
    fig.update_xaxes(title_text='X Frequency', title_standoff=5, showticklabels=False, row=1, col=2)
    fig.update_yaxes(title_text='Y Pixel', title_standoff=5, showticklabels=False, row=1, col=1)
    fig.update_yaxes(title_text='Y Frequency', title_standoff=5, showticklabels=False, row=1, col=2)

    # pio.write_html(fig, file="./data/amplitude.html", auto_play=True)

    # Build the figure
    return fig

@st.cache_data
def create_lena_fft():
    '''
    Creates a plotly figure containing the image and corresponding fft for Lena
    '''
    img = cv2.imread("./data/lena.jpg", 0) # read in image as a grayscale
    fft_img = np.log(np.abs(np.fft.fftshift(np.fft.fft2(img)))) # compute 2d fft

    seq = np.array([img / 255, fft_img / np.max(fft_img)]) # concatenate images

    # create fig and add titles
    fig = px.imshow(seq, color_continuous_scale='gray', facet_col=0, facet_col_spacing=0.02, height=500)
    fig.layout.annotations[0]['text'] = "Image in Spatial Domain"
    fig.layout.annotations[1]['text'] = "Image in Frequency Domain"

    # remove axes
    fig.update_layout(coloraxis_showscale=False)
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)

    return fig

def create_fft_showcase(option):
    '''
    step-by-step visualization of the fft process
    '''

    img = select_shape(option)# create 5x5 matrix of grayscale pixels

    fig = make_subplots(2, 2, row_heights=[100]*2, column_widths=[100]*2)

    # create fft after each step
    # fft1 stores the frequency magnitudes of img in the y direction
    # fft2 stores the frequency magnitudes of fft1 in the x direction
    # ftt3 stores the shifted and log'd copy of fft2
    fft1, fft2, fft3 = np.empty_like(img), np.empty_like(img), None

    for col in range(img.shape[1]): # compute y direction ffts
        fft1[:, col] = np.fft.fft(img[:, col])

    for row in range(fft1.shape[0]): # compute x direction ffts
        fft2[row, :] = np.fft.fft(fft1[row, :])
    
    fft3 = np.log(np.abs(np.fft.fftshift(fft2.copy()))) # shift and log

    # create heat maps
    h1 = go.Heatmap({'z': np.around(img, 2)}, colorscale='Viridis', texttemplate="%{z}", showscale=False)
    h2 = go.Heatmap({'z': np.around(np.abs(fft1), 2)}, colorscale='Viridis', texttemplate="%{z}", showscale=False)
    h3 = go.Heatmap({'z': np.around(np.abs(fft2), 2)}, colorscale='Viridis', texttemplate="%{z}", showscale=False)
    h4 = go.Heatmap({'z': np.around(fft3, 2)}, colorscale='Viridis', texttemplate="%{z}", showscale=False)

    fig.add_trace(h1, row=1, col=1)
    fig.add_trace(h2, row=1, col=2)
    fig.add_trace(h3, row=2, col=1)
    fig.add_trace(h4, row=2, col=2)

    # update yaxes so they are not upside down
    fig.update_yaxes(autorange="reversed", showticklabels=False)

    return fig

@st.cache_data
def create_kspace():
    '''
    Creates a visualization showing the transformation of MRI image from from raw kspace data
    into a spatial image 
    '''

    slice_kspace = np.load("./data/knee_kspace.npy")[:-1] # import data (first 20 2d slice of the kspace in 5 slice intervals)

    frames = np.empty(shape=(slice_kspace.shape[0], 2, slice_kspace.shape[-2], slice_kspace.shape[-1]))
    frames[:, 0, :, :] = np.log((np.abs(slice_kspace) + 1e-9)) # put original kspace data into the frames (log magnitude)
    for i in range(slice_kspace.shape[0]):
        fft_img = np.log(np.abs(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(slice_kspace[i]))))) # compute inverse fft
        frames[i, 1, :, :] = normalize(fft_img) # add to frame
        frames[i, 0, :, :] = normalize(frames[i, 0, :, :])
    
    fig = px.imshow(frames, color_continuous_scale='gray', animation_frame=0, facet_col=1, height=500, binary_compression_level=5) # create animation
    
    fig.layout.annotations[0]['text'] = "Original Kspace Image"
    fig.layout.annotations[1]['text'] = "Reconstructed Spatial Image"
    
    # update plot title for each frame
    _ = [fig.frames[i]['layout'].update(title_text=f'Slice {i*5}') for i in range(slice_kspace.shape[0])]

    # update slider and figure
    fig.update_layout(sliders=[{'currentvalue': {'visible' : False}}], coloraxis_showscale=False) 
    
    # update figure axes
    fig.update_xaxes(title_text='Kx', title_standoff=5, showticklabels=False, row=1, col=1)
    fig.update_yaxes(title_text='Ky', title_standoff=5, showticklabels=False, row=1, col=1)
    fig.update_xaxes(title_text='X Pixel', title_standoff=5, showticklabels=False, row=1, col=2)
    fig.update_yaxes(title_text='Y Pixel', title_standoff=5, showticklabels=False, row=1, col=2)

    return fig

@st.cache_data
def normalize(x): # normalize data values in an array to the range [0 - 1]
    return x - np.min(x) / (np.max(x) - np.min(x))


def select_shape(option): # returns a 5x5 array of a shape based on the option string

    if option == None:
        return (np.zeros(shape=(5,5)) + 0.5)
    
    elif option == 'Star':
        return np.array([[0.5, 0.5, 1, 0.5, 0.5],
                         [0.5, 1, 1, 1, 0.5],
                         [1, 1, 1, 1, 1],
                         [0.5, 1, 0.5, 1, 0.5], 
                         [1, 0.5, 0.5, 0.5, 1]])
    
    elif option == 'Square':
        return np.array([[0.5, 0.5, 0.5, 0.5, 0.5],
                         [0.5, 1, 1, 1, 0.5],
                         [0.5, 1, 1, 1, 0.5],
                         [0.5, 1, 1, 1, 0.5], 
                         [0.5, 0.5, 0.5, 0.5, 0.5]])

    elif option == 'Circle':
        return np.array([[0.5, 0.5, 1, 0.5, 0.5],
                         [0.5, 1, 1, 1, 0.5],
                         [1, 1, 1, 1, 1],
                         [0.5, 1, 1, 1, 0.5], 
                         [0.5, 0.5, 1, 0.5, 0.5]]) 

    elif option == 'X':
        return np.array([[1, 0.5, 0.5, 0.5, 1],
                         [0.5, 1, 0.5, 1, 0.5],
                         [0.5, 0.5, 1, 0.5, 0.5],
                         [0.5, 1, 0.5, 1, 0.5], 
                         [1, 0.5, 0.5, 0.5, 1]])               


def create_mri_reconstruction(image_num, radius):
    image = cv2.imread(f"./data/Tumor_{image_num}.JPG", 0) # read in image in gray scale

    center_x, center_y = (image.shape[0]) / 2, (image.shape[1]) / 2 # get image center pixel

    # create x and y coordinates for each pixel
    x = np.arange(-center_x, center_x, 1)
    y = np.arange(-center_y, center_y, 1)
    xx, yy = np.meshgrid(y, x)

    # create mask
    mask = np.where(xx**2 + yy**2 <= radius**2, 0, 1)

    # compute fft and reconstruct the image after masking
    fft_img = np.fft.fftshift(np.fft.fft2(image))*mask
    reconstructed_img = np.log(np.abs(np.fft.ifft2(np.fft.ifftshift(fft_img))))
    reconstructed_img = np.where(reconstructed_img > 3.25, reconstructed_img*3, 0)
    
    fig_fft = px.imshow(np.log(np.abs(fft_img)), color_continuous_scale='viridis')
    fig_reconstruct = px.imshow(reconstructed_img, color_continuous_scale='gray')

    # remove color axis
    fig_fft.update_layout(coloraxis_showscale=False) 
    fig_reconstruct.update_layout(coloraxis_showscale=False) 
    
    # update figure axes
    fig_fft.update_xaxes(showticklabels=False)
    fig_fft.update_yaxes(showticklabels=False)
    fig_reconstruct.update_xaxes(showticklabels=False)
    fig_reconstruct.update_yaxes(showticklabels=False)


    # return both the masked fft and the reconstructed image
    return image, fig_fft, fig_reconstruct 
