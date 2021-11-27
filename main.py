import warnings
import geopandas as gpd
#import descartes
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from PIL import Image 

#page metadata => lifted directly from Aaron's Previous Group
st.set_page_config(page_title="You're a Winner! Formulating Campaign Strategies to Ensure Candidaet Winnability",
    layout="wide"
    )
warnings.filterwarnings("ignore")
st.set_option("deprecation.showPyplotGlobalUse",False)

# importing local csvs 
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
    st.subheader('by Data Science Fellowship Cohort 8 - Group 2')
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

    pol_ad = Image.open('assets/pol_ad.jpg')
    
    #Not sure if we should keep the columns imo
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown(
            "1. **Does a candidate have to be social media-savvy?**"
        )
        st.markdown(    
            "2. **Do I listen to electoral surveys?**"
        )
        st.markdown(    
            "3. **Does money really buy happiness, or in this case a seat in the office?**"
        )
        st.markdown(    
            "4. **Should a political campaign focus on the younger voters? Perhaps the middle-aged ones?**"
        )
        st.markdown(    
            "5. **Should a candidate focus on specific regions or municipalities?**"
        )
    with col2:
        st.image(pol_ad, caption='Source: BBC')

def data_method():
    #3rd page - Data Sources and Methodology
    st.title('Data Sources and Methodology')
    st.subheader("Data Sources")
    st.write("")
    data_sources = Image.open("assets/Data_source.JPG")
    st.image(data_sources)
    st.write("")
    st.subheader("Methodology")
    methodology = Image.open("assets/Methodology.JPG")
    st.image(methodology)
    
def win_loss():
    # 4th page - Winners vs Losers
    # we can do yung select box thingy + multiple ifs para isang page nalang expenditures saka socmed
    option = st.selectbox('Select from the ff:', ['Social Media Presence', 'Contributions', 'Expenditures'])
    if option == 'Social Media Presence':
        st.markdown('The Philippines ranked first in internet and social media usage last 2020')
        st.markdown('The average Filipino is on social media for around 3 hours and 50 minutes daily')
        # add photo nalang siguro or columns formatting
    
    elif option == 'Contributions':
        #Data Loading
        df_raw = pd.read_csv('UBALDO/data/2019-candidate-campaigns.csv', index_col=0)
        df_raw.fillna(0, inplace=True)
        
        #Data Wrangling
        candidates = df_raw.iloc[:, 0:3]
        contrib = df_raw.loc[:, 'Win': 'Total Contributions Received'].fillna(0)
        df = pd.concat([candidates,contrib], axis =1)
        df['Total Cash Contributions'] = df['Cash Contributions Received from Other Sources'] + df['Cash Contributions Received from Political Party']
        df['Total In-Kind Contributions'] = df['In-Kind Contributions Received from Other Sources'] + df['In-Kind Contributions Received from Political Party']
        
        #Insight
        st.markdown('Winners have received significant contributions both in Cash and In-Kind.') 
        st.markdown('This could mean that they have enough resources to fund for all their expenditures.')
        
        #Visual stored in fig
        win = df.iloc[:, :4]
        total_contrib = df.iloc[:, 8:-1]
        df1 = pd.concat([win, total_contrib], axis=1)
        melted_df1=pd.melt(df1.iloc[:, 3:], id_vars="Win")
        
        fig = plt.figure(figsize=(12,8), dpi = 150)
        sns.boxplot(y = melted_df1['variable'],
            x = melted_df1['value'],
            hue = melted_df1['Win'])
        
        plt.ylabel('Contribution Sources', fontsize=12)
        plt.xlabel('Amount of Contributions received', fontsize=12)
        plt.xticks(fontsize=15)

        #display graph
        st.pyplot(fig)
        # add photo nalang siguro or columns formatting
    
    elif option == 'Expenditures':
        st.markdown('Winners significantly spent on political ads vs those who lost the election.')
        st.markdown('Other expenses of winning candidates are travel expenses, compensation of campaigners, and below-the-line materials vs those who lost the election.')

def profile():
    # 5th page - Voter profiling
    # same as above, para yung kmeans saka geospatial plot isang page nalang din
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import geopandas as gpd

    #read shapefile
    shapefile = gpd.read_file('data/Provinces/Provinces.shp')
    shapefile["x"] = shapefile.geometry.centroid.x
    shapefile["y"] = shapefile.geometry.centroid.y

    #read csv
    voter = pd.read_csv('data/2016-2019-voter-data.csv')

    #data manip
    province = {}
    for i in voter["Province"].unique(): 
        s_province = [x for x in shapefile["PROVINCE"].unique() if i == x.upper()]
        if len(s_province) == 1:
            province[i] = s_province[0]
        else:
            province[i] = 'INPUT'      
    #Manually inserting Province
    province['NCR'] = 'Metropolitan Manila'
    province['DAVAO OCCIDENTAL'] = 'Shariff Kabunsuan'
    # Replace province name
    voter["Province"] = voter["Province"].replace(province)
    #dropping unnecessary columns
    voter = voter.loc[:,['Region','Province','Municipality','2016-Registered_Voters','2019-Total_Voters_Turnout']]

    #data wrangling
    #sum per province
    province_data = voter.groupby("Province").agg({'2016-Registered_Voters':'sum','2019-Total_Voters_Turnout':'mean'}).reset_index()
    province_data ['2016-Registered_Voters'] = province_data ['2016-Registered_Voters']/1000000
    # merging shapefile and province data
    merged_data = pd.merge(shapefile, province_data, left_on = 'PROVINCE', right_on = 'Province')

    #Plot 1
    variable0 = "2016-Registered_Voters"
    vmin0, vmax0 = merged_data["2016-Registered_Voters"].min(), merged_data["2016-Registered_Voters"].max()
    fig, axes = plt.subplots(1, figsize=(15, 10))
    axes.set_title("2016 Registered Voters (in million)", size = 18)
    merged_data.plot(column=variable0, cmap='OrRd', linewidth=0.8, ax=axes, edgecolor='0.8', vmin=vmin0, vmax=vmax0)
    sm1 = plt.cm.ScalarMappable(cmap='OrRd', norm=plt.Normalize(vmin=vmin0, vmax=vmax0))
    cbar = fig.colorbar(sm1,ax=axes) # geomap

    #Dataframe 1 (top5)
    province_data.rename(columns = {'2016-Registered_Voters': '2016 Registered Voters (in million)'}, inplace= True)
    province_data.sort_values(by='2016 Registered Voters (in million)', ascending=False, inplace = True)
    df= province_data.set_index('Province').head(5)
    pd.DataFrame(df['2016 Registered Voters (in million)']) 

    #Plot 2
    variable0 = "2019-Total_Voters_Turnout"
    vmin0, vmax0 = merged_data["2019-Total_Voters_Turnout"].min(), merged_data["2019-Total_Voters_Turnout"].max()
    fig, axes = plt.subplots(1, figsize=(15, 10))
    axes.set_title("2019 Total Voters Turnout", size = 18)
    merged_data.plot(column=variable0, cmap='OrRd', linewidth=0.8, ax=axes, edgecolor='0.8', vmin=vmin0, vmax=vmax0)
    sm1 = plt.cm.ScalarMappable(cmap='OrRd', norm=plt.Normalize(vmin=vmin0, vmax=vmax0))
    cbar = fig.colorbar(sm1,ax=axes)

    #Dataframe 2 (top5)
    province_data.rename(columns = {'2019-Total_Voters_Turnout': '2019 Total Voters Turnout (%)'}, inplace= True)
    province_data.sort_values(by='2019 Total Voters Turnout (%)', ascending=False, inplace = True)
    df1= province_data.set_index('Province').head(5)
    pd.DataFrame(df1['2019 Total Voters Turnout (%)'])
    pass

def conclusions():
    # 6th page - Conclusions and recommendations
    pass

def references():
    # 7th page - References
    pass


# loop to run pages
page = st.sidebar.radio('Page selection:', list_of_pages)

if page == "The Project":
    project()
    
elif page == "Background and Objective":
    background()
    
elif page == "Data Sources and Methodology":
    data_method()

elif page == "Winners vs Losers":
    win_loss()
    
elif page == "Voter Profiling":
    profile()

elif page == "Conclusion and Recommendations":
    conclusions()

elif page == "References":
    references()