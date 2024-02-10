import os
import sys
import subprocess
from moviepy.editor import VideoFileClip, AudioFileClip
import streamlit as st
from pydub import AudioSegment
from pydub.silence import split_on_silence
import speech_recognition as sr
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from heapq import nlargest

r1 = sr.Recognizer()


def audio_extraction(video_file, output_ext="wav"):
    """Converts video to audio using Moviepy which uses ffmpeg"""
    # splitting the input video file name into the original name and the extension
    filename, ext = os.path.splitext(video_file)
    clip = VideoFileClip(video_file)
    return clip.audio.write_audiofile(f"{filename}.{output_ext}", codec="libvorbis")

@st.cache_data
def transcribe_large_audio(path):
    sound = AudioSegment.from_file(path)
    chunks = split_on_silence(sound, min_silence_len=700, silence_thresh=sound.dBFS - 14, keep_silence=700)  # sepa
    folder_name = "audio-chunks"
    if not os.path.isdir(folder_name):
        os.mkdir(folder_name)
    whole_text = ""
    for i, audio_chunk in enumerate(chunks, start=1):
        chunk_filename = os.path.join(folder_name, f"chunk{i}.flac")
        audio_chunk.export(chunk_filename, format="flac")

        with sr.AudioFile(chunk_filename) as source:
            # r1.adjust_for_ambient_noise(source, duration = 0.5)
            audio_listened = r1.record(source)
            try:
                text = r1.recognize_google(audio_listened)
            except sr.UnknownValueError as e:
                print("Error:", str(e))
            else:
                text = f"{text.capitalize()}. "
                print(chunk_filename, ":", text)
                whole_text += text
    return whole_text


def transcribe_without_streamlit(path):
    sound = AudioSegment.from_file(path)
    chunks = split_on_silence(sound, min_silence_len=700, silence_thresh=sound.dBFS - 14, keep_silence=700)  # sepa
    folder_name = "audio-chunks"
    if not os.path.isdir(folder_name):
        os.mkdir(folder_name)
    whole_text = ""
    for i, audio_chunk in enumerate(chunks, start=1):
        chunk_filename = os.path.join(folder_name, f"chunk{i}.flac")
        audio_chunk.export(chunk_filename, format="flac")

        with sr.AudioFile(chunk_filename) as source:
            # r1.adjust_for_ambient_noise(source, duration = 0.5)
            audio_listened = r1.record(source)
            try:
                text = r1.recognize_google(audio_listened)
            except sr.UnknownValueError as e:
                print("Error:", str(e))
            else:
                text = f"{text.capitalize()}. "
                print(chunk_filename, ":", text)
                whole_text += text
    return whole_text


def summarize(text, per):
    nlp = spacy.load('en_core_web_lg')
    doc = nlp(text)
    tokens=[token.text for token in doc] # filter out common words and punctuations
    word_frequencies = {}   # segment the text into words, punctuation according to grammatical rules
    for word in doc:
        if word.text.lower() not in list(STOP_WORDS):
            if word.text.lower() not in punctuation:
                if word.text not in word_frequencies.keys():
                    word_frequencies[word.text] = 1
                else:
                    word_frequencies[word.text] += 1
    max_frequency=max(word_frequencies.values())
    for word in word_frequencies.keys():  # normalizing the count
        word_frequencies[word]=word_frequencies[word]/max_frequency
    sentence_tokens= [sent for sent in doc.sents]
    sentence_scores = {}
    for sent in sentence_tokens:  # sum of the normalized count for each sentence
        for word in sent:
            if word.text.lower() in word_frequencies.keys():
                if sent not in sentence_scores.keys():
                    sentence_scores[sent]=word_frequencies[word.text.lower()]
                else:
                    sentence_scores[sent]+=word_frequencies[word.text.lower()]
    select_length=int(len(sentence_tokens)*per)
    # selecting a percentage of the highest ranked sentences
    summary = nlargest(select_length, sentence_scores,key=sentence_scores.get)
    final_summary = [word.text for word in summary]
    summary = ''.join(final_summary)
    return summary