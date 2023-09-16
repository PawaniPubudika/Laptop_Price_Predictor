import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle


data = pd.read_csv('/Users/pawanipubudika/Documents/ready/laptop price predictor/data/preprocessed_data.csv')
st.title('Exploratory Data Analysis')
sidebar = st.sidebar
st.title('Data Summary')
st.write(data.head(10))

st.title('Data Visualizations')
st.sidebar.header("Exploring Laptop Market Trends")
st.header("Exploring Laptop Market Trends")
selected_variable = st.sidebar.selectbox("Select Variable", ['Company', 'TypeName', 'Ram', 'OpSys','TouchScreen','IPS','CPU_name','ClockSpeed','Weight','PPI','HDD','SSD','Gpu brand'])
plot_type = st.sidebar.radio("Select Plot Type", ['Count Plot', 'Bar Plot'])

if plot_type == 'Count Plot':
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.write("<span style='font-size:26px;'>Count Plot Analysis</span>", unsafe_allow_html=True)
    plt.figure(figsize=(10, 6))
    sns.countplot(data=data, x=selected_variable, palette='plasma')
    plt.xticks(rotation='vertical')
    st.pyplot()
elif plot_type == 'Bar Plot':
    st.write("<span style='font-size:26px;'>Bar Plot Analysis</span>", unsafe_allow_html=True)
    plt.figure(figsize=(15, 7))
    sns.barplot(x=data[selected_variable], y=data['Price'])
    plt.xticks(rotation='vertical')
    st.pyplot()

st.header("Understanding Price Distributions in Laptop Sales")
st.sidebar.header("Understanding Price Distributions in Laptop Sales")
plot_type = st.sidebar.radio("Select Plot Type", ['Price Distribution', 'Log(Price) Distribution'])
log_price = np.log(data['Price'])
if plot_type == 'Price Distribution':
    st.write("<span style='font-size:26px;'>Price Distribution</span>", unsafe_allow_html=True)
    plt.figure(figsize=(6, 4))
    sns.distplot(data['Price'], color='purple')
    plt.xlabel('Price')
    plt.ylabel('Density')
    plt.title('Distribution Plot of Price')
    st.pyplot()
elif plot_type == 'Log(Price) Distribution':
    st.write("<span style='font-size:26px;'>Log(Price) Distribution</span>", unsafe_allow_html=True)
    plt.figure(figsize=(6, 4))
    sns.distplot(log_price, color='purple')
    plt.xlabel('Log(Price)')
    plt.ylabel('Density')
    plt.title('Distribution Plot of LogExploring Relationships: Correlation Heatmap of Laptop Specifications(Price)')
    st.pyplot()


st.sidebar.header("Correlation Heatmap")
show_plot = st.sidebar.checkbox("Show Plot")
if show_plot:
    st.header("Exploring Relationships: Correlation Heatmap of Laptop Specifications")
    columns_of_interest = ['Ram', 'Weight', 'TouchScreen', 'IPS', 'PPI', 'ClockSpeed', 'HDD', 'SSD']
    correlation_matrix = data[columns_of_interest].corr()
    plt.figure(figsize=(10, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap among Specified Columns')
    st.pyplot()
    
    
st.sidebar.header("Final Output")
show_image = st.sidebar.checkbox('Show Image')

if show_image:
    st.header("Final Prediction on the whole Dataset")
    st.image('/Users/pawanipubudika/Documents/ready/laptop price predictor/src/output.png', caption='Caption for the image', use_column_width=True)


