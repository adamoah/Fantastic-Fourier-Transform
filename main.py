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
    st.title("Hello World")
    # value = st.slider("Brightness", 0, 10, value=1, step=1)
    fig = load()
    st.plotly_chart(fig)


if __name__ == "__main__":
    main()