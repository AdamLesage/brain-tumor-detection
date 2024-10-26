import streamlit as st
from sklearn.datasets import load_wine
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Title
st.title('Data Analysis 📊')

# Header
st.header('This section is to show what I can do with data analysis and Streamlit.')

st.write('Dataset used here is Wine dataset from sklearn.datasets')

# Load the dataset
wine = load_wine()
wine_data = pd.DataFrame(data=wine.data, columns=wine.feature_names)
wine_data['target'] = wine.target

# Display the dataset
st.write(wine_data)

# Display donut chart
median_alcohol = wine_data['alcohol'].median()

# Filter the wine data to get rows where alcohol content is above the median
above_median_alcohol = wine_data[wine_data['alcohol'] > median_alcohol]

# Calculate the mean of alcohol content
mean_alcohol = wine_data['alcohol'].mean()

# Filter the wine data to get rows where alcohol content is above the mean
above_mean_alcohol_count = wine_data['alcohol'] > mean_alcohol

# Display the mean of alcohol content
st.subheader(f"Mean alcohol content: {mean_alcohol:.2f}")

# Create a donut chart for above mean alcohol content
fig, ax = plt.subplots()
ax.pie([above_mean_alcohol_count.sum(), len(wine_data) - above_mean_alcohol_count.sum()], labels=['Above Mean', 'Below Mean'], autopct='%1.1f%%', startangle=90)
ax.axis('equal')
st.pyplot(fig)

# Additional Data Analysis

# Histogram of alcohol content
st.subheader('Histogram of Alcohol Content')
fig, ax = plt.subplots()
sns.histplot(wine_data['alcohol'], kde=True, ax=ax)
st.pyplot(fig)
gs

# Correlation heatmap
st.subheader('Correlation Heatmap')
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(wine_data.corr(), annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
st.pyplot(fig)
