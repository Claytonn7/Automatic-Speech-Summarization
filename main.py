import streamlit as st
import os
from pydub import AudioSegment
from pydub.silence import split_on_silence
import speech_recognition as sr
import numpy as np
import librosa, librosa.display
import matplotlib.pyplot as plt
import functions as f


backgroundColor = "green"
xy = 0
summary = ""
def disable():
    st.session_state.disabled = True


def enable():
    if "disabled" in st.session_state and st.session_state.disabled == True:
        st.session_state.disabled = False
        print('true')


def textfile(text):
    f1 = open("MyFile1.txt", "w+")
    f1.write(text)
    f1.close()
    return f1

if 'stage' not in st.session_state:
    st.session_state.stage = 0

def set_stage(stage):
    st.session_state.stage = stage




if "disabled" not in st.session_state:
    st.session_state.disabled = False


st.title("Automatic Speech Summarization")


add_selectbox = st.sidebar.selectbox(
    'How would you like to be contacted?',
    ('Email', 'Home phone', 'Mobile phone')
)

uploaded_file = st.file_uploader("Choose an audio or video file", type=["mp4", "flac"])

if uploaded_file is not None:
    result = f.transcribe_large_audio(uploaded_file)
    print(result)
    slider_number = st.slider("Select percentage of summarization",
                              min_value=0.1,
                              max_value=0.5,
                              value=0.2,
                              step=0.01)
    col1, col2, col3 = st.columns([1, 1, 1])  # Adjust column ratios as needed

    with col2:
        #if st.button("Generate Summary"):
        x = st.button("Generate Summary", on_click=set_stage, args=(1,))
        # x = st.button("Generate Summary", on_click=disable, disabled=st.session_state.disabled)

    if st.session_state.stage > 0:
        summary = f.summarize(result, slider_number)

        st.write("Summary : ")
        st.write(summary)

        st.session_state.stage = 0