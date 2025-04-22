import numpy as np
import cv2
import streamlit as st
import plotly.express as px
import pandas as pd
import time


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

    



    
