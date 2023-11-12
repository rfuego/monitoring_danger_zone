import streamlit as st
import cv2 
import os 
  
st.set_page_config(layout="wide")
st.title('Проект команды "Fuego"') 
frame_placeholder = st.empty()


frame = cv2.imread('images/fon.jpg')

frame_placeholder.image(frame, channels="RGB")