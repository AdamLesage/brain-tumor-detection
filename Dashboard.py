import streamlit as st

# Title
st.title('Streamlit App to show my skills 🧠')

# Header
st.header('Streamlit App for [Butterfl.ai](http://www.butterfl.ai) Interview 🦋')

# Introduction
st.subheader('🚀 Introduction')
st.write("""
    Welcome to my first experience with Streamlit! This app links to a former project, 
    which is a CNN model to detect brain tumors from MRI images.
""")

# Project Demo Section
st.subheader('📊 Project Demo')
st.write("""
    See the CNN model in action and predict the type of tumor in the **Project Demo** section in the sidebar.
""")

# About Me Section
st.subheader('👤 About Me')
st.write("""
    Learn more about me in the **About Me** section in the sidebar.
""")