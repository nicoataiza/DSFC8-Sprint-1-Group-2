   
#import warnings
import geopandas as gpd
#import descartes
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
#from PIL import Image => Check if this is in requirements.txt

#page metadata => lifted directly from Aaron's Previous Group
st.set_page_config(page_title="You're a Winner! Formulating Campaign Strategies to Ensure Candidaet Winnability",
    layout="wide"
    )
warnings.filterwarnings("ignore")
st.set_option("deprecation.showPyplotGlobalUse",False)

#importing local csvs 
# INSERT HERE

#list of pages ("AKA OUTLINE")
list_of_pages = [
    "The Project",
    "Background and Objective",
    "Data Sources and Methodology",
    "Winners vs Losers",
    "Voter Profiling",
    "Conclusion and Recommendations",
    "References"
]

#NTS => beta columns = css.flexbox/css.grid
def project():
    #1st page - Project details
    st.title("You're a Winner! Formulating Campaign Strategies to Ensure Candidaet Winnability")
    st.subheader('by Data Science Fellowship Cohort 9 - Group 2')
    st.write('Adrian, Bono, Grace, MaCris, Nico, Sofia (mentored by Aaron)')

    #teacher_image = Image.open('teacher.jpg') 

    """
    col1, col2 = st.beta_columns(2)
    with col1:
        st.image(
            teacher_image,
            caption='A teacher with her class of 59 students in a Quezon City public school. Source: The Guardian'
        )
    with col2:
        st.markdown(
            "In this **exploratory data analysis**, we aim to uncover the distribution of public education resources "
            "across the Philippines and identify critical deficiencies "
            "through an assessment of **Maintenance and Other Operating Expenses (MOOE)** "
            "allocation in the different regions."
        )
    """

def background():
    #2nd page - Background of Study and Research Questions
    st.title('Background and Objective')
    st.markdown(
       "With the upcoming 2022 elections, our group wanted to review past election data and determine what did the \
        **winning candidates do (or not do) to set themselves apart from the rest.**")

    st.write(
        "With respect to this, the group primarly on four questions derived from existing election data: "
    )

    #sdg4_image = Image.open('sdg4.jpg')
    
    #Not sure if we should keep the columns imo
    col1, col2 = st.beta_columns([1, 2])
    with col1:
        st.markdown(
            "1. **Does a candidate have to be social media-savvy?**"
        )
        st.markdown(    
            "2. **Does money really buy happiness, or in this case a seat in the office?**"
        )
        st.markdown(    
            "3. **Should a political campaign focus on the younger voters? Perhaps the middle-aged ones?**"
        )
        st.markdown(    
            "4. **Should a candidate focus on specific regions or municipalities?**"
        )
    with col2:
        st.image(sdg4_image, caption='Source: Think Sustainability')

def data_method():
    #3rd page - Data Sources and Methodology
    st.title('Data Sources and Methodology')
    st.write("")
    data_sources = Image.open("assets/Data_source.JPG")
    st.image(data_sources)
    st.write("")
    methodology = Image.open("assets/Methodology.JPG")
    st.image(methodology)