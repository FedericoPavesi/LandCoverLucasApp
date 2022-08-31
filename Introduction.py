import streamlit as st

def Introduction():
    st.markdown('# Introduction')
    st.sidebar.markdown('# Introduction')
    
def DatabaseCreation():
    st.markdown('# Database creation')
    st.sidebar.markdown('# Database creation')
    
def AlgorithmsTraining():
    st.markdown('# Algorithms training')
    st.sidebar.markdown('# Algorithms training')
    
def MapClassification():
    st.markdown('# Map classification')
    st.sidebar.markdown('# Map classification')
    
page_names_to_funcs = {'Introduction' : Introduction,
                       'Database creation' : DatabaseCreation,
                       'Algorithms training' : AlgorithmsTraining,
                       'Map classification' : MapClassification}



#selected_page = st.sidebar.selectbox('Select a page', page_names_to_funcs.keys())
#page_names_to_funcs[selected_page]()

st.markdown('# Introduction') 

st.markdown('In this work we will analyse an approach to classify land-cover composition of any region employing [Copernicus Sentinel-2](https://sentinels.copernicus.eu/web/sentinel/missions/sentinel-2) detections and [Eurostat Lucas points](https://ec.europa.eu/eurostat/web/lucas).')

st.markdown('_If you are interested in codes used to perform the process you can look up [here](https://github.com/FedericoPavesi/Lucas_points_for_Sentinel2_LandCover_Download)._')

st.markdown('\n')

st.markdown('Lucas points database provides a collection of geo-referenced points with an associated three-digits land-cover class. At the same time, Copernicus Sentinel-2 satellites costellation provides a wide collection of high-resolution earth surface images, composed by 12 spectral bands collecting different wavelenght reflectance.')

st.markdown('In following discussion you will find a detailed explanation of each step performed in order to first obtain a database of pixels reflectance with an associated land cover class, then to train proper land cover classifiers using an algorithm from machine learning (Random Forest) and one from deep learning (Multi-Layer Perceptron), and finally to apply trained classifiers to a map of a regional reflectance.')


