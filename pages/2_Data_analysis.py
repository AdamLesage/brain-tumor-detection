import streamlit as st
from sklearn.datasets import load_wine
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Title
st.title('Data Analysis 📊')

# Header
st.header('This section is to show what I can do with data analysis and Streamlit.')

st.write('Dataset used here is Wine dataset from sklearn.datasets')

# Load the dataset
wine = load_wine()
df = pd.DataFrame(wine.data, columns=wine.feature_names)

# Display the dataset
st.write('Displaying the dataset below')

# Sidebar
st.sidebar.title('Data Analysis')
st.sidebar.header('Select the type of plot you want to see')
plot_type = st.sidebar.multiselect(
    "select the type of plot",
    ('Scatter Plot', 'Histogram', 'Box Plot', 'Correlation Plot'),
    default=['Scatter Plot', 'Histogram', 'Box Plot', 'Correlation Plot']
)

# Scatter Plot
if 'Scatter Plot' in plot_type:
    st.subheader('Scatter Plot')
    fig = px.scatter(df, x='alcohol', y='malic_acid')
    st.plotly_chart(fig)

# Histogram
if 'Histogram' in plot_type:
    st.subheader('Histogram')
    fig = px.histogram(df, x='alcohol')
    st.plotly_chart(fig)

# Box Plot
if 'Box Plot' in plot_type:
    st.subheader('Box Plot')
    fig = px.box(df, x='alcohol', y='malic_acid')
    st.plotly_chart(fig)

# Correlation Plot
if 'Correlation Plot' in plot_type:
    st.subheader('Correlation Plot')
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)
