import geopandas as gpd
# import descartes
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import warnings
from PIL import Image

# page metadata => lifted directly from Aaron's Previous Group
st.set_page_config(page_title="You're a Winner! Formulating Campaign Strategies to Ensure Candidaet Winnability",
                   layout="wide"
                   )
warnings.filterwarnings("ignore")
st.set_option("deprecation.showPyplotGlobalUse", False)

# importing local csvs 
# INSERT HERE

# list of pages ("AKA OUTLINE")
list_of_pages = [
    "The Project",
    "Background and Objective",
    "Data Sources and Methodology",
    "Winners vs Losers",
    "Voter Profiling",
    "Conclusion and Recommendations",
    "References"
]


# NTS => beta columns = css.flexbox/css.grid
def project():
    # 1st page - Project details
    st.title("You're a Winner! Formulating Campaign Strategies to Ensure Candidaet Winnability")
    st.subheader('by Data Science Fellowship Cohort 8 - Group 2')
    st.write('Adrian, Bono, Grace, MaCris, Nico, Sofia (mentored by Aaron)')

    # teacher_image = Image.open('teacher.jpg')

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
    # 2nd page - Background of Study and Research Questions
    st.title('Background and Objective')
    st.markdown(
        "With the upcoming 2022 elections, our group wanted to review past election data and determine what did the \
         **winning candidates do (or not do) to set themselves apart from the rest.**")

    st.write(
        "With respect to this, the group primarly on five questions derived from existing election data: "
    )

    pol_ad = Image.open('assets/pol_ad.jpg')
    st.write("")
    # Not sure if we should keep the columns imo
    col1, col2 = st.beta_columns([1, 2])
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
    # 3rd page - Data Sources and Methodology
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
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
        import plotly.express as px

        # read data from local
        campaign = pd.read_csv('2019-candidate-campaigns.csv')
        votes = pd.read_csv('2019-senatorial-votes.csv')
        # merge data
        df_merged = pd.merge(left=campaign, right=votes, how='left', on='Candidate')
        df_merged.head()
        # count NaNs
        # meron din replace nalang with 0 haha
        df_merged.fillna(0, inplace=True)
        df_merged.isna().sum()
        # add features
        ## total social media interactions
        df_merged['Twitter - Total Interactions'] = (df_merged['Twitter - Number of Mentions'] + df_merged[
            'Twitter - Number of Users'] + df_merged['Twitter- Total Favorites'] + df_merged[
                                                         'Twitter - Total Retweets'] + + df_merged[
            'Twitter - Total Replies']) / 1000000
        df_merged['Facebook - Total Interactions'] = (df_merged['Facebook - Number of Candidate Posts'] + df_merged[
            'Facebook - Total Comments'] + df_merged['Facebook - Total Shares'] + df_merged[
                                                          'Facebook - Total Reactions']) / 1000000
        df_merged['Total Social Media Interactions'] = df_merged['Twitter - Total Interactions'] + df_merged[
            'Facebook - Total Interactions']

        ## win-loss categorical
        def win_category(x):
            if x == 1:
                return 'Win'
            else:
                return 'Lose'

        df_merged['Win-Category'] = df_merged['Win'].apply(lambda x: win_category(x))
        ## sort by votes
        df_merged.sort_values(by='Votes', ascending=False, inplace=True)

        df_merged.head()
        # split into winners & losers
        df_winner = df_merged[df_merged['Win'] == 1].copy()
        df_loser = df_merged[df_merged['Win'] == 0].copy()
        # sort by votes
        df_winner.sort_values(by='Votes', inplace=True, ascending=False)
        df_loser.sort_values(by='Votes', inplace=True, ascending=False)
        # plotly winners
        fig = px.bar(data_frame=df_winner, x='Candidate', y='Votes')
        fig.show()
        # plot losers
        fig = px.bar(data_frame=df_loser, x='Candidate', y='Votes')
        fig.show()
        # plot all with color hue
        fig = px.bar(data_frame=df_merged, x='Candidate', y='Votes', color='Win-Category')
        fig.show()
        # scatter socmed activities vs votes
        # hover_data > tooltips hover
        # color > similar to hue from seaborn
        # size > column based, need to know if can be set to static value
        fig = px.scatter(data_frame=df_merged, x='Total Social Media Interactions', y='Votes', color='Win-Category',
                         hover_data=['Candidate'], size='Votes')
        fig.show()
        # plotly boxplot
        fig = px.box(data_frame=df_merged, y='Total Social Media Interactions', x='Win-Category')
        fig.show()
        # try side by side plotting
        import plotly.graph_objects as go
        candidates = df_merged['Candidate']
        fig = go.Figure(data=[go.Bar(name='Social Media Activity', x=candidates,
                                     y=df_merged['Total Social Media Interactions'] * 1000000),
                              go.Bar(name='Votes', x=candidates, y=df_merged['Votes'])])
        fig.update_layout(barmode='group')
        fig.show()
        # scale variables bago i-plot side by side haha

        # scatter activity + wins
        plt.figure(figsize=(10, 10))
        sns.scatterplot(y='Votes', x='Total Social Media Interactions', data=df_merged, hue='Win-Category', s=150)
        sns.despine()
        plt.title('Votes vs. Social Media Interactions', size=18)
        plt.xlabel('Social Media Interactions - in millions', size=12)
        plt.ylabel('Votes', size=12)
        # plt.show()
        plt.savefig("scatter-votesXsocmed.png", transparent=True, bbox_inches='tight')

        # barplot of votes
        plt.figure(figsize=(15, 5))
        sns.barplot(x='Candidate', y='Votes', data=df_merged.head(30), hue='Win-Category')
        sns.despine()
        plt.title('Senatorial Election Results', size=18)
        plt.ylabel('Votes', size=12)
        plt.xticks(rotation=90)
        # plt.show()
        plt.savefig("barplot-votes.png", transparent=True, bbox_inches='tight')

        # barplot of SMS activites
        plt.figure(figsize=(15, 5))
        sns.barplot(x='Candidate', y='Total Social Media Interactions', data=df_merged.head(30), hue='Win-Category')
        sns.despine()
        plt.title('Social Media Presence per Candidate', size=18)
        plt.ylabel('Social Media Interactions - in millions', size=12)
        plt.xticks(rotation=90)
        # plt.show()
        plt.savefig("barplot-socmed.png", transparent=True, bbox_inches='tight')

        # box plot of social media activities per winners and losers
        plt.figure(figsize=(10, 10))
        sns.boxplot(x='Win-Category', y='Total Social Media Interactions', data=df_merged)
        plt.title('Social Media Presence per Candidate', size=18)
        plt.ylabel('Social Media Interactions - in millions', size=12)
        plt.show()
        # add photo nalang siguro or columns formatting

    elif option == 'Contributions':
        # Data Loading
        df_raw = pd.read_csv('UBALDO/data/2019-candidate-campaigns.csv', index_col=0)
        df_raw.fillna(0, inplace=True)

        # Data Wrangling
        candidates = df_raw.iloc[:, 0:3]
        contrib = df_raw.loc[:, 'Win': 'Total Contributions Received'].fillna(0)
        df = pd.concat([candidates, contrib], axis=1)
        df['Total Cash Contributions'] = df['Cash Contributions Received from Other Sources'] + df[
            'Cash Contributions Received from Political Party']
        df['Total In-Kind Contributions'] = df['In-Kind Contributions Received from Other Sources'] + df[
            'In-Kind Contributions Received from Political Party']

        # Insight
        st.markdown('Winners have received significant contributions both in Cash and In-Kind.')
        st.markdown('This could mean that they have enough resources to fund for all their expenditures.')

        # Visual stored in fig
        win = df.iloc[:, :4]
        total_contrib = df.iloc[:, 8:]
        df1 = pd.concat([win, total_contrib], axis=1)
        melted_df1 = pd.melt(df1.iloc[:, 3:], id_vars="Win")

        fig = plt.figure(figsize=(12, 8), dpi=150)
        sns.boxplot(y=melted_df1['variable'],
                    x=melted_df1['value'],
                    hue=melted_df1['Win'])

        plt.title('Boxplot of Contributions Received - Winners vs. Losers', fontsize=20)
        plt.ylabel('Contribution Sources', fontsize=12)
        plt.xlabel('Amount of Contributions received', fontsize=12)
        plt.xticks(fontsize=15)

        # display graph
        st.pyplot(fig)
        # add photo nalang siguro or columns formatting

    elif option == 'Expenditures':
        st.markdown('Winners significantly spent on political ads vs those who lost the election.')
        st.markdown(
            'Other expenses of winning candidates are travel expenses, compensation of campaigners, and below-the-line materials vs those who lost the election.')
        # LOAD Data
        df_cluster = pd.read_csv('ABIERA/data/2019-candidate-campaigns.csv', index_col=0)
        df_cluster = df_cluster.fillna(0)

        st.markdown('Winners significantly spent on political ads vs those who lost the election.')
        st.markdown(
            'Other expenses of winning candidates are travel expenses, compensation of campaigners, and below-the-line materials vs those who lost the election.')

        pd.set_option('display.float_format', '{:,.3f}'.format)
        from sklearn.preprocessing import Normalizer
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score

        feature_cols = [
            'Travel Expenses',
            'Compensation of campaigners, etc.',
            'Communications',
            'Stationery, Printing, and Distribution',
            'Employment of Poll Watchers',
            'Rent, Maintenance, etc.',
            'Political Meetings and Rallies',
            'Pol Ads'
        ]
        df_cluster = df_cluster[feature_cols]
        X = df_cluster[feature_cols]
        X = Normalizer().fit_transform(X.values)
        # find KMeans
        kmeans = KMeans(n_clusters=3)
        kmeans.fit(X)
        y_kmeans = kmeans.predict(X)
        # silhouette score
        s_score = silhouette_score(X, y_kmeans)
        # plotting for elbow curve and silhouette score
        inertia = []
        sil = []
        for k in range(2, 10):
            km = KMeans(n_clusters=k, random_state=1)
            km.fit(X)
            y_pred = km.predict(X)

            inertia.append((k, km.inertia_))
            sil.append((k, silhouette_score(X, y_pred)))
        # show figure
        fig, ax = plt.subplots(1, 2, figsize=(16, 8))
        # Plotting Elbow Curve
        x_iner = [x[0] for x in inertia]
        y_iner = [x[1] for x in inertia]
        ax[0].plot(x_iner, y_iner)
        ax[0].set_xlabel('Number of Clusters')
        ax[0].set_ylabel('Intertia')
        ax[0].set_title("Inertia Score - AKA. 'Elbow Curve'")

        # Plotting Silhouetter Score
        x_sil = [x[0] for x in sil]
        y_sil = [x[1] for x in sil]
        ax[1].plot(x_sil, y_sil)
        ax[1].set_xlabel('Number of Clusters')
        ax[1].set_ylabel('Silhouetter Score')
        ax[1].set_title('Silhouette Score Curve')
        # show figure
        st.markdown('We determine the best number of clusters by finding the inertia and silhoette score')
        st.pyplot(fig)

        # Change index to cluster label
        # kmeans = KMeans(n_clusters=3)
        # kmeans.fit(X)
        # labels = kmeans.predict(X)
        # feature_cols = ['Travel Expenses',
        # 'Compensation of campaigners, etc.', 'Communications',
        # 'Stationery, Printing, and Distribution', 'Employment of Poll Watchers',
        # 'Rent, Maintenance, etc.', 'Political Meetings and Rallies', 'Pol Ads','Win']
        # df = df[feature_cols]
        # df['Cluster Labels'] = labels
        # df = df.set_index('Cluster Labels')
        # df = df.groupby("Cluster Labels").mean().reset_index()
        # df["Total"] = df["Travel Expenses"] + df["Compensation of campaigners, etc."]+df["Communications"]+df["Stationery, Printing, and Distribution"]+df["Employment of Poll Watchers"]+df["Rent, Maintenance, etc."]+df["Political Meetings and Rallies"]+df["Pol Ads"]


# showtable
# df
        #Boxplots of Expenditures
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        pd.set_option('display.float_format', '{:,.4f}'.format)
        pd.set_option('max_columns', 100)
        df1 = pd.read_csv(r"data\2019-campaign-spending_clean.csv")
        df2 = pd.read_csv(r"data\2019_Votes_PulseAsiaSurvey.csv")
        
        win = df1.iloc[:, :3]
        exp = df1.iloc[:, 14:22]
        melted_df1=pd.melt(df1.iloc[:, 2:], id_vars="Win")
        melted_df1
        
        melted_df2=pd.melt(df1.iloc[:, 2:], id_vars="Win")

        plt.figure(figsize=(12,8), dpi = 150)
        sns.boxplot(y = melted_df2['variable'],
                    x = melted_df2['value'],
                    hue = melted_df2['Win'])

        plt.title('Boxplots of Individual Expenditure Item of Winners vs Losers', fontsize = 20)
        plt.ylabel('', fontsize= 40)
        plt.xlabel('Expenditure Amount (in Ten Millions)', fontsize=20)
        plt.xticks( fontsize=12)

        plt.savefig("All expenses.png", dpi=150, bbox_inches="tight")
        plt.show()
        
        #Scatterplots of Electoral Surveys vs Votes
        plt.figure(figsize=(10,8))
        sns.scatterplot(x= "Votes", y = "PulseAsia Survey 2019 (Jan 26-31)", data = df2, hue = "Win")
        plt.title("Electoral Survey (Jan 26-31, 2019) vs Actual Votes", fontsize=20)
        plt.xlabel("Actual Votes")
        plt.ylabel("Electoral Survey %")
        plt.show()
        
        plt.figure(figsize=(10,8))
        sns.scatterplot(x= "Votes", y = "PulseAsia Survey 2019 (Feb 24-28)", data = df2, hue = "Win")
        plt.title("Electoral Survey (Feb 24-28, 2019) vs Actual Votes", fontsize=20)
        plt.xlabel("Actual Votes")
        plt.ylabel("Electoral Survey %")
        plt.show()
        
        plt.figure(figsize=(10,8))
        sns.scatterplot(x= "Votes", y = "PulseAsia Survey 2019 (Mar 23-27)", data = df2, hue = "Win")
        plt.title("Electoral Survey (Mar 23-27, 2019) vs Actual Votes", fontsize=20)
        plt.xlabel("Actual Votes")
        plt.ylabel("Electoral Survey %")
        plt.show()

        plt.figure(figsize=(10,8))
        sns.scatterplot(x= "Votes", y = "PulseAsia Survey 2019 (Apr 10-14)", data = df2, hue = "Win")
        plt.title("Electoral Survey (Apr 10-14, 2019) vs Actual Votes", fontsize=20)
        plt.xlabel("Actual Votes")
        plt.ylabel("Electoral Survey %")
        plt.show()
        
        plt.figure(figsize=(10,8))
        sns.scatterplot(x= "Votes", y = "PulseAsia Survey 2019 (May 3-6)", data = df2, hue = "Win")
        plt.title("Electoral Survey (May 3-6, 2019) vs Actual Votes", fontsize=20)
        plt.xlabel("Actual Votes")
        plt.ylabel("Electoral Survey %")
        plt.show()

        #Correlations of Votes and Surveys
        df2.corr()

def profile():
    # 5th page - Voter profiling
    # same as above, para yung kmeans saka geospatial plot isang page nalang din
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import geopandas as gpd

    # read shapefile
    shapefile = gpd.read_file('data/Provinces/Provinces.shp')
    shapefile["x"] = shapefile.geometry.centroid.x
    shapefile["y"] = shapefile.geometry.centroid.y

    # read csv
    voter = pd.read_csv('data/2016-2019-voter-data.csv')

    # data manip
    province = {}
    for i in voter["Province"].unique():
        s_province = [x for x in shapefile["PROVINCE"].unique() if i == x.upper()]
        if len(s_province) == 1:
            province[i] = s_province[0]
        else:
            province[i] = 'INPUT'
            # Manually inserting Province
    province['NCR'] = 'Metropolitan Manila'
    province['DAVAO OCCIDENTAL'] = 'Shariff Kabunsuan'
    # Replace province name
    voter["Province"] = voter["Province"].replace(province)
    # dropping unnecessary columns
    voter = voter.loc[:, ['Region', 'Province', 'Municipality', '2019-Registered_Voters', '2019-Total_Voters_Turnout']]

    # data wrangling
    # sum per province
    province_data = voter.groupby("Province").agg(
        {'2019-Registered_Voters': 'sum', '2019-Total_Voters_Turnout': 'mean'}).reset_index()
    province_data['2019-Registered_Voters'] = province_data['2019-Registered_Voters'] / 1000000
    # merging shapefile and province data
    merged_data = pd.merge(shapefile, province_data, left_on='PROVINCE', right_on='Province')

    # Plot 1
    variable0 = "2019-Registered_Voters"
    vmin0, vmax0 = merged_data["2019-Registered_Voters"].min(), merged_data["2019-Registered_Voters"].max()
    fig, axes = plt.subplots(1, figsize=(15, 10))
    axes.set_title("2019 Registered Voters (in million)", size=18)
    merged_data.plot(column=variable0, cmap='OrRd', linewidth=0.8, ax=axes, edgecolor='0.8', vmin=vmin0, vmax=vmax0)
    sm1 = plt.cm.ScalarMappable(cmap='OrRd', norm=plt.Normalize(vmin=vmin0, vmax=vmax0))
    cbar = fig.colorbar(sm1, ax=axes)

    # Dataframe 1 (top5)
    province_data.rename(columns={'2019-Registered_Voters': '2019 Registered Voters (in million)'}, inplace=True)
    province_data.sort_values(by='2019 Registered Voters (in million)', ascending=False, inplace=True)
    df = province_data.set_index('Province').head(5)
    print(pd.DataFrame(df['2019 Registered Voters (in million)']))

    # Plot 2
    variable0 = "2019-Total_Voters_Turnout"
    vmin0, vmax0 = merged_data["2019-Total_Voters_Turnout"].min(), merged_data["2019-Total_Voters_Turnout"].max()
    fig, axes = plt.subplots(1, figsize=(15, 10))
    axes.set_title("2019 Total Voters Turnout", size=18)
    merged_data.plot(column=variable0, cmap='OrRd', linewidth=0.8, ax=axes, edgecolor='0.8', vmin=vmin0, vmax=vmax0)
    sm1 = plt.cm.ScalarMappable(cmap='OrRd', norm=plt.Normalize(vmin=vmin0, vmax=vmax0))
    cbar = fig.colorbar(sm1, ax=axes)

    # Dataframe 2 (top5)
    province_data.rename(columns={'2019-Total_Voters_Turnout': '2019 Total Voters Turnout (%)'}, inplace=True)
    province_data.sort_values(by='2019 Total Voters Turnout (%)', ascending=False, inplace=True)
    df1 = province_data.set_index('Province').head(5)
    print(pd.DataFrame(df1['2019 Total Voters Turnout (%)']))


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
