import numpy as np
import streamlit as st
import cv2 as cv
import matplotlib.pyplot as plt
from functions.test import func

def main():
    '''
    main function displays the webside
    '''
    st.title("Hello World")
    st.write(func())


if __name__ == "__main__":
    main()