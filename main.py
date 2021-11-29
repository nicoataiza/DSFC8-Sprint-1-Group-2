# import descartes
import geopandas as gpd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
import numpy as np
import numpy as np
import numpy as np
import pandas as pd
import pandas as pd
import pandas as pd
import pandas as pd
import plotly.express as px
import seaborn as sns
import seaborn as sns
import seaborn as sns
import streamlit as st
import warnings
from PIL import Image
from io import BytesIO
import squarify
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import Normalizer

pd.set_option('max_columns', 100)
pd.set_option('display.float_format', '{:,.3f}'.format)

# Set page config
st.set_page_config(
    page_title="You're a Winner! Formulating Campaign Strategies to Ensure Candidate Winnability",
    layout="wide"
)
warnings.filterwarnings("ignore")
st.set_option("deprecation.showPyplotGlobalUse", False)

# List pages
list_of_pages = [
    "The Project",
    "Background and Objective",
    "Data Sources and Methodology",
    "Winners vs Losers",
    "Voter Profiling",
    "Conclusion and Recommendations",
    "References"
]


# List all functions
# @st.cache(allow_output_mutation=True)
def load_data(option=None):
    if option == "2019-campaigns":
        data = pd.read_csv('JOVES/data/2019-candidate-campaigns.csv')
    elif option == "2019-senatorial-votes":
        data = pd.read_csv('JOVES/data/2019-senatorial-votes.csv')
    elif option == "2019-campaigns-v2":
        data = pd.read_csv('UBALDO/data/2019-candidate-campaigns.csv', index_col=0)
    elif option == "2019-campaigns-v3":
        data = pd.read_csv('ABIERA/data/2019-candidate-campaigns.csv', index_col=0)
    elif option == "2019-campaigns-v4":
        data = pd.read_csv("data/2019-campaign-spending_clean.csv")
    elif option == "2019-surveys":
        data = pd.read_csv("data/2019_Votes_PulseAsiaSurvey.csv")
    elif option == "provinces":
        data = gpd.read_file('data/Provinces/Provinces.shp')
    elif option == "2016-2019-voters":
        data = pd.read_csv('data/2016-2019-voter-data.csv')

    return data


def project():
    st.title("You're a Winner! Formulating Campaign Strategies to Ensure Candidate Winnability")
    st.subheader('by Data Science Fellowship Cohort 8 - Group 2')
    st.write('Adrian, Bono, Grace, MaCris, Nico, Sofia (mentored by Aaron)')

    election_img = Image.open('JOVES/media/ph_election.jpg')

    col1, col2 = st.beta_columns(2)
    with col1:
        st.image(
            election_img,
            caption='image from Rappler - How can we improve PH democracy?'
        )
    with col2:
        st.markdown(
            "In this **exploratory data analysis**, we aim to identify the strategies that every candidate "
            "participating in this upcoming election should look into "
            "through an assessment of previous election data including contributions received, expenditures, "
            "voter profiles and social media presence."
        )


def background():
    st.title('Background and Objective')
    st.markdown(
        "With the upcoming 2022 elections, our group wanted to review past election data and determine what did the \
         **winning candidates do (or not do) to set themselves apart from the rest.**")

    st.write(
        "With respect to this, the group primarly on five questions derived from existing election data: "
    )

    pol_ad = Image.open('assets/pol_ad.jpg')
    st.write("")

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
    st.title('Data Sources and Methodology')
    st.write("")
    # col1, col2 = st.beta_columns(2)
    # with col1:
    # data_sources = Image.open("assets/Data Source.JPG")
    # st.image(data_sources)
    # with col2:
    # methodology = Image.open("assets/Methodology3.JPG")
    # st.image(methodology)

    #     st.write("")
    #     data_sources = Image.open("assets/Data Source.jpg")
    #     st.image(data_sources)

    #     st.write("")
    #     methodology = Image.open("assets/Methodology3.jpg")
    #     st.image(methodology)

    st.write("")
    st.subheader("Data Sources")
    data_sources = Image.open("assets/Data_source.JPG")
    st.image(data_sources)

    st.write("")
    st.subheader("Methodology")
    methodology = Image.open("assets/Methodology.JPG")
    st.image(methodology)


def win_loss():
    st.title("What did Winners do differently?")
    option = st.selectbox('Analysis on:',
                          ['Social Media Presence', 'Electoral Surveys', 'Expenditures', 'Contributions'])

    if option == 'Social Media Presence':

        col1, col2 = st.beta_columns([5, 11])
        with col1:
            st.markdown('The Philippines ranked first in internet and social media usage last 2020.')
            st.markdown('The average Filipino is on social media for around 3 hours and 50 minutes daily.')
            st.markdown('And yet, there is a weak correlation between the number votes and total social media interaction.')
            st.subheader('Social Media presence while impactful is not a guarantee to winning votes.')

        with col2:
            # Load the data
            campaign = load_data(option="2019-campaigns")
            votes = load_data(option="2019-senatorial-votes")

            # Merge the data
            df_merged = pd.merge(left=campaign, right=votes, how='left', on='Candidate')

            # Fill nulls
            df_merged.fillna(0, inplace=True)

            # Perform aggregations
            df_merged['Twitter - Total Interactions'] = (
                                                                df_merged['Twitter - Number of Mentions']
                                                                + df_merged['Twitter - Number of Users']
                                                                + df_merged['Twitter- Total Favorites']
                                                                + df_merged['Twitter - Total Retweets']
                                                                + df_merged['Twitter - Total Replies']
                                                        ) / 1000000
            df_merged['Facebook - Total Interactions'] = (
                                                                 df_merged['Facebook - Number of Candidate Posts']
                                                                 + df_merged['Facebook - Total Comments']
                                                                 + df_merged['Facebook - Total Shares']
                                                                 + df_merged['Facebook - Total Reactions']
                                                         ) / 1000000
            df_merged['Total Social Media Interactions'] = df_merged['Twitter - Total Interactions'] \
                                                           + df_merged['Facebook - Total Interactions']

            # Generate mapping function
            def win_category(x):
                if x == 1:
                    return 'Win'
                else:
                    return 'Lose'

            # Map function: 1 to Win and 0 to Lose
            df_merged['Win-Category'] = df_merged['Win'].apply(lambda x: win_category(x))

            # Sort the data
            df_merged.sort_values(by='Votes', ascending=False, inplace=True)

            # Generate scatter plot
            fig = plt.figure(figsize=(10, 6), dpi=150)
            sns.scatterplot(y='Votes', x='Total Social Media Interactions', data=df_merged, hue='Win-Category', s=150)
            sns.despine()
            plt.title('Votes vs. Social Media Interactions', size=18)
            plt.xlabel('Social Media Interactions - in millions', size=12)
            plt.ylabel('Votes', size=12)
            st.pyplot(fig)

    elif option == 'Electoral Surveys':

        st.subheader(
            "It is wise for the candidates to track their survey results as it is highly correlated with actual votes.")

        # Load the data
        df2 = load_data(option="2019-surveys")

        col1, col2 = st.beta_columns([10, 10])
        with col1:
            # Compute correlations of votes and surveys
            votes = df2.iloc[:, 4:5]
            pulse = df2.iloc[:, 6:11]
            df_cor = pd.concat([votes, pulse], axis=1)
            st.table(df_cor.corr())

        with col2:
            option = st.selectbox(
                'Select Date:',
                ['Jan 26-31, 2019', 'Feb 24-28, 2019', 'Mar 23-27, 2019', 'Apr 10-14, 2019', 'May 3-6, 2019'])

            # Generate mapping function
            def win_category(x):
                if x == 1:
                    return 'Win'
                else:
                    return 'Lose'

            # Map function: 1 to Win and 0 to Lose
            df2['Win Category'] = df2['Win'].apply(lambda x: win_category(x))

            # Plot scatter plots of electoral surveys vs votes
            if option == "Jan 26-31, 2019":
                fig1 = plt.figure(figsize=(10, 6))
                sns.scatterplot(x="Votes", y="PulseAsia Survey 2019 (Jan 26-31)", data=df2, hue="Win Category")
                plt.title("Electoral Survey (Jan 26-31, 2019) vs Actual Votes", fontsize=20)
                plt.xlabel("Actual Votes")
                plt.ylabel("Electoral Survey %")
                st.pyplot(fig1)

            elif option == "Feb 24-28, 2019":
                fig2 = plt.figure(figsize=(10, 6))
                sns.scatterplot(x="Votes", y="PulseAsia Survey 2019 (Feb 24-28)", data=df2, hue="Win Category")
                plt.title("Electoral Survey (Feb 24-28, 2019) vs Actual Votes", fontsize=20)
                plt.xlabel("Actual Votes")
                plt.ylabel("Electoral Survey %")
                st.pyplot(fig2)

            elif option == "Mar 23-27, 2019":
                fig3 = plt.figure(figsize=(10, 6))
                sns.scatterplot(x="Votes", y="PulseAsia Survey 2019 (Mar 23-27)", data=df2, hue="Win Category")
                plt.title("Electoral Survey (Mar 23-27, 2019) vs Actual Votes", fontsize=20)
                plt.xlabel("Actual Votes")
                plt.ylabel("Electoral Survey %")
                st.pyplot(fig3)

            elif option == "Apr 10-14, 2019":
                fig4 = plt.figure(figsize=(10, 6))
                sns.scatterplot(x="Votes", y="PulseAsia Survey 2019 (Apr 10-14)", data=df2, hue="Win Category")
                plt.title("Electoral Survey (Apr 10-14, 2019) vs Actual Votes", fontsize=20)
                plt.xlabel("Actual Votes")
                plt.ylabel("Electoral Survey %")
                st.pyplot(fig4)

            elif option == "May 3-6, 2019":
                fig5 = plt.figure(figsize=(10, 6))
                sns.scatterplot(x="Votes", y="PulseAsia Survey 2019 (May 3-6)", data=df2, hue="Win Category")
                plt.title("Electoral Survey (May 3-6, 2019) vs Actual Votes", fontsize=20)
                plt.xlabel("Actual Votes")
                plt.ylabel("Electoral Survey %")
                st.pyplot(fig5)




    elif option == 'Contributions':
        # Load the data
        df_raw = load_data(option="2019-campaigns-v2")

        # Fill nulls
        df_raw.fillna(0, inplace=True)

        # Clean the data
        candidates = df_raw.iloc[:, 0:3]
        contrib = df_raw.loc[:, 'Win': 'Total Contributions Received'].fillna(0)
        df = pd.concat([candidates, contrib], axis=1)

        # Perform aggregations
        df['Total Cash Contributions'] = df['Cash Contributions Received from Other Sources'] \
                                         + df['Cash Contributions Received from Political Party']
        df['Total In-Kind Contributions'] = df['In-Kind Contributions Received from Other Sources'] \
                                            + df['In-Kind Contributions Received from Political Party']

        col1, col2 = st.beta_columns([5, 11])
        with col1:
            st.markdown('Winners have received significant contributions both in cash and in-Kind.')
            st.markdown('This could mean that they have enough resources to fund for all their expenditures.')
            st.markdown(
                'Thus, enabling them to have more opportunities to in terms of their campaign spending'
                'and as we have seen from Expenditure analysis has impacted the winnability of a candidate.'
            )

        with col2:
            # Clean the data
            win = df.iloc[:, :4]
            total_contrib = df.iloc[:, 8:]
            df1 = pd.concat([win, total_contrib], axis=1)
            melted_df1 = pd.melt(df1.iloc[:, 3:], id_vars="Win")

            # Generate mapping function
            def win_category(x):
                if x == 1:
                    return 'Win'
                else:
                    return 'Lose'

            # Map function: 1 to Win and 0 to Lose
            melted_df1['Win Category'] = melted_df1['Win'].apply(lambda x: win_category(x))

            # Generate box plots
            fig = plt.figure(figsize=(12, 8), dpi=150)
            my_pal = {"Win": "#1f77b4", "Lose": "orange"}
            sns.boxplot(y=melted_df1['variable'],
                        x=melted_df1['value'],
                        hue=melted_df1['Win Category'],
                        palette=my_pal)

            # plt.title('Boxplot of Contributions Received - Winners vs. Losers', fontsize=20)
            plt.legend(loc='lower right')
            plt.ylabel('Contribution Sources', fontsize=12)
            plt.xlabel('Amount of Contributions Received', fontsize=12)
            plt.xticks(fontsize=15)

            # Display graph
            st.pyplot(fig)

    elif option == 'Expenditures':
        st.subheader('Initial Analysis on Expenditures by Item')
        st.markdown('')
        st.markdown('')

        col1, col2 = st.beta_columns([5, 11])
        with col1:
            st.markdown('Winners significantly spent on political ads vs those who lost the election.')
            st.markdown(
                'Other expenses of winning candidates are travel expenses, compensation of campaigners,'
                'and below-the-line materials vs those who lost the election.'
            )

        with col2:
            # Generate box plots of expenditures

            # Load the data
            df = load_data(option="2019-campaigns-v4")

            win = df.iloc[:, :3]
            exp = df.iloc[:, 14:22]

            df1 = pd.concat([win, exp], axis=1)

            melted_df1 = pd.melt(df1.iloc[:, 2:], id_vars="Win")

            # Generate mapping function
            def win_category(x):
                if x == 1:
                    return 'Win'
                else:
                    return 'Lose'

            # Map function: 1 to Win and 0 to Lose
            melted_df1['Win Category'] = melted_df1['Win'].apply(lambda x: win_category(x))

            # Plot the data
            fig = plt.figure(figsize=(12, 8), dpi=150)
            sns.boxplot(y=melted_df1['variable'],
                        x=melted_df1['value'],
                        hue=melted_df1['Win Category'])

            # plt.title('Boxplots of Individual Expenditure Item of Winners vs Losers', fontsize=20)
            plt.ylabel('', fontsize=40)
            plt.xlabel('Expenditure Amount', fontsize=15)
            plt.xticks(fontsize=12)

            st.pyplot(fig)

        # Expenditures Clustering
        st.write('\n')
        st.markdown('')
        st.subheader('Now we try to cluster candidates based on their expenditures.')
        st.markdown('')

        col1, col2 = st.beta_columns([11, 5])
        with col1:
            # Load the data
            df_cluster = load_data(option="2019-campaigns-v3")
            df1 = df_cluster

            # Fill nulls
            df_cluster = df_cluster.fillna(0)

            # Clean the data
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

            # Normalize the data
            X = Normalizer().fit_transform(X.values)

            # Implement k-Means clustering
            kmeans = KMeans(n_clusters=3)
            kmeans.fit(X)
            y_kmeans = kmeans.predict(X)

            # Compute silhoutte score
            s_score = silhouette_score(X, y_kmeans)

            # Plot elbow curve and silhouette score
            inertia = []
            sil = []
            for k in range(2, 10):
                km = KMeans(n_clusters=k, random_state=1)
                km.fit(X)
                y_pred = km.predict(X)

                inertia.append((k, km.inertia_))
                sil.append((k, silhouette_score(X, y_pred)))

            # Show figure
            fig, ax = plt.subplots(1, 2, figsize=(16, 8), dpi=150)

            # Plot elbow curve
            x_iner = [x[0] for x in inertia]
            y_iner = [x[1] for x in inertia]
            ax[0].plot(x_iner, y_iner)
            ax[0].set_xlabel('Number of Clusters')
            ax[0].set_ylabel('Intertia')
            ax[0].set_title("Inertia Score - AKA. 'Elbow Curve'")

            # Plot silhouette score
            x_sil = [x[0] for x in sil]
            y_sil = [x[1] for x in sil]
            ax[1].plot(x_sil, y_sil)
            ax[1].set_xlabel('Number of Clusters')
            ax[1].set_ylabel('Silhouette Score')
            ax[1].set_title('Silhouette Score Curve')

            st.pyplot(fig)

        with col2:
            st.markdown('We determine the best number of clusters by finding the inertia and silhouette score')
            st.markdown('Using the Elbow Method, we have identified that the optimal number of clusters is 3.')

        st.markdown('')
        st.markdown('')
        st.subheader("After implementing clustering, we profiled the clusters based on their individual expenses.")
        
        kmeans = KMeans(n_clusters=3)
        kmeans.fit(X)
        labels = kmeans.predict(X)
        
        #set cluster Labels as index
        df_temp = df_cluster
        df_temp["Win"] = df1.loc[:, "Win"]
        df_temp['Cluster'] = labels
        df_temp = df_temp.set_index('Cluster')
        df_temp = df_temp.groupby("Cluster").mean().reset_index()
        
        col1, col2 = st.beta_columns([5,11])
        #generate summary table
        with col1:
            st.markdown('')
            st.markdown(
                'As we can see from the table below, the cluster with the highest expenditure of PHP 72.7 million' 
                ' also has the highest rate of winning which is at 42.31%' 
            )
            
            st.markdown('')
            df_temp['Total Expenditure'] = df_temp.sum(axis=1)
            df_temp = df_temp[["Cluster","Total Expenditure", "Win"]]
            st.write(df_temp)
            
            st.markdown('')
            st.markdown('Thus, it can be said that higher expenditure corresponds to higher chances of winning')
            
        # Regenerate k-Means clustering
        kmeans = KMeans(n_clusters=3)
        kmeans.fit(X)
        labels = kmeans.predict(X)
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
        df_cluster['Cluster Labels'] = labels
        df_cluster = df_cluster.groupby("Cluster Labels").mean().reset_index()

        # Profile each cluster
        with col2:
            st.markdown('')
            column = st.selectbox('Choose an expenditure item:', feature_cols)

            #for column in df_cluster.columns:
            if column == "index" or column == "Cluster Labels":
                pass
            else:
                fig, ax = plt.subplots(1, 1, figsize=(12, 8), dpi=150)
                sns.barplot(
                    x="Cluster Labels",
                    y=column,
                    hue="Cluster Labels",
                    data=df_cluster,
                    dodge=False
                )
                plt.ticklabel_format(style='plain', axis='y', useOffset=False)
                ax.set(xlabel='Clusters', ylabel='Expenses (in pesos)', title=column)
                ax.legend([], [], frameon=False)
                st.pyplot(fig)


def profile():
    option = st.selectbox('Voter Pofile by:', ['Age Range', 'Registered Voters'])

    if option == 'Age Range':
        st.header("Age Range of Registered Voters in 2016")
        # Data Wrangling
        voter = load_data(option="2016-2019-voters")
        voter = voter[['2016-Registered_17-19', '2016-Registered_20-24',
       '2016-Registered_25-29', '2016-Registered_30-34',
       '2016-Registered_35-39', '2016-Registered_40-44',
       '2016-Registered_45-49', '2016-Registered_50-54',
       '2016-Registered_55-59', '2016-Registered_60-64',
       '2016-Registered_65-Above']]
        df1 = pd.DataFrame(voter.sum())
        df1 = df1.reset_index()
        df1.rename(columns={0:"Votes"},inplace=True)

        #tree map
        plt.figure(figsize=(16,8))
        perc = [f'{i/df1["Votes"].sum()*100:5.2f}%' for i in df1['Votes']]
        lbl = [f'{el[0][-5:]} = {el[1]}' if el[0][-5:]!= "Above" else f"65-Above = {el[1]}" for el in zip(df1['index'], perc)]
        squarify.plot(sizes=df1["Votes"], label=lbl, alpha=.8,linewidth=2.5)
        plt.axis("off")
        sns.set(font_scale=2.5)
        #plt.title("Age Distribution of Registered Voters in 2016",fontsize=36)
        st.pyplot(plt)
    elif option == 'Registered Voters':

        st.header("Voter Profile by Registered Voters")
        # Read shapefile
        shapefile = load_data(option="provinces")
        shapefile["x"] = shapefile.geometry.centroid.x
        shapefile["y"] = shapefile.geometry.centroid.y

        # Read csv
        voter = load_data(option="2016-2019-voters")

        # Wrangle the data
        province = {}
        for i in voter["Province"].unique():
            s_province = [x for x in shapefile["PROVINCE"].unique() if i == x.upper()]
            if len(s_province) == 1:
                province[i] = s_province[0]
            else:
                province[i] = 'INPUT'

        # Manually insert province
        province['NCR'] = 'Metropolitan Manila'
        province['DAVAO OCCIDENTAL'] = 'Shariff Kabunsuan'

        # Replace province name
        voter["Province"] = voter["Province"].replace(province)

        # Drop unnecessary columns
        voter = voter.loc[:,
                ['Region', 'Province', 'Municipality', '2019-Registered_Voters', '2019-Total_Voters_Turnout']]

        # Get sum per province
        province_data = voter.groupby(
            "Province"
        ).agg(
            {'2019-Registered_Voters': 'sum', '2019-Total_Voters_Turnout': 'mean'}
        ).reset_index()
        province_data['2019-Registered_Voters'] = province_data['2019-Registered_Voters'] / 1000000

        # Merge shapefile and province data
        merged_data = pd.merge(shapefile, province_data, left_on='PROVINCE', right_on='Province')

        col1, col2 = st.beta_columns([8, 9])
        with col1:
            st.subheader('Manila, Cebu and Cavite have the highest registered voters.')
            st.markdown('The same provinces is also the top 3 Provinces in terms population count')
            st.markdown('')
            st.markdown('')

            # Get top 5
            province_data.rename(columns={'2019-Registered_Voters': '2019 Registered Voters (in million)'},
                                inplace=True)
            province_data.sort_values(by='2019 Registered Voters (in million)', ascending=False, inplace=True)
            df = province_data.set_index('Province').head(5)
            # print(pd.DataFrame(df['2019 Registered Voters (in million)']))
            st.write(province_data.set_index('Province').iloc[:, :1].head(5))

        with col2:
            # Plot 1
            variable0 = "2019-Registered_Voters"
            vmin0, vmax0 = merged_data["2019-Registered_Voters"].min(), merged_data["2019-Registered_Voters"].max()
            fig, axes = plt.subplots(1, figsize=(7, 8))
            axes.set_title("2019 Registered Voters (in millions)", size=12)
            merged_data.plot(column=variable0, cmap='OrRd', linewidth=0.8, ax=axes, edgecolor='0.8', vmin=vmin0,
                            vmax=vmax0)
            sm1 = plt.cm.ScalarMappable(cmap='OrRd', norm=plt.Normalize(vmin=vmin0, vmax=vmax0))
            cbar = fig.colorbar(sm1, ax=axes)
            st.pyplot(fig)


#             buf = BytesIO()
#             fig.savefig(buf, format="png")
#             st.image(buf)

# Plot 2
#         variable0 = "2019-Total_Voters_Turnout"
#         vmin0, vmax0 = merged_data["2019-Total_Voters_Turnout"].min(), merged_data["2019-Total_Voters_Turnout"].max()
#         fig, axes = plt.subplots(1, figsize=(15, 10))
#         axes.set_title("2019 Total Voters Turnout", size=18)
#         merged_data.plot(column=variable0, cmap='OrRd', linewidth=0.8, ax=axes, edgecolor='0.8', vmin=vmin0, vmax=vmax0)
#         sm1 = plt.cm.ScalarMappable(cmap='OrRd', norm=plt.Normalize(vmin=vmin0, vmax=vmax0))
#         cbar = fig.colorbar(sm1, ax=axes)

#         # Get top 5
#         province_data.rename(columns={'2019-Total_Voters_Turnout': '2019 Total Voters Turnout (%)'}, inplace=True)
#         province_data.sort_values(by='2019 Total Voters Turnout (%)', ascending=False, inplace=True)
#         df1 = province_data.set_index('Province').head(5)
#         print(pd.DataFrame(df1['2019 Total Voters Turnout (%)']))


def conclusions():
    # 6th page - Conclusions and recommendations
    st.title('Conclusions and Recommendations')
    st.write(
        "The election is more than just a simple popularity game. Though  popular candidates have a head start, it does not"                         " necessarily translate to votes. Election campaigning is a strategic game that combines both machineries and resources"
        " as well as careful planning and execution to win the votes of the people.")

    st.write("As such, here are the factors that candidate should focus on developing a winning campaign strategy.")
    st.write(
        "**SOCIAL MEDIA.**  Investing in social media does not necessarily equate to winning. Candidates must not focus on this aspect only.")
    st.write(
        "**POLL.** Candidates must listen to survey and adjust their strategy as necessary.")
    st.write(
        "**MONEY.** Since money fuels campaign and has significant impact on the election result, candidate must take into consideration their financial capacity.")
    st.write(
        "**TARGET.** Age and areas must be considered for strategizing their campaign.")
    data_sources = Image.open("assets/recommendations.jpg")
    st.image(data_sources)

    pass


def references():
    # 7th page - References
    st.title('References')
    st.write(
        "•   Census-based Population Projections in collaboration with the Inter-Agency Working Group on Population Projections. (2010)."            " [online] Available at: https://psa.gov.ph/sites/default/files/attachments/hsd/pressrelease/Table4_9.pdf.")
    st.write(
        "•   psa.gov.ph. (n.d.). Philippine Statistics Authority | Republic of the Philippines. [online] Available"
        " at: https://psa.gov.ph/gender-stat/wmf.")
    st.write(
        "•   psa.gov.ph. (n.d.). Highlights of the Philippine Population 2020 Census of Population and Housing (2020 CPH) | Philippine"              " Statistics Authority. [online] Available at: https://psa.gov.ph/content/highlights-philippine-population-2020-census-population-and-housing-2020-cph.")
    st.write(
        "•   Chua, K. (2021). PH remains top in social media, internet usage worldwide – report. [online] Rappler. Available at:"                   "https://www.rappler.com/technology/internet-culture/hootsuite-we-are-social-2021-philippines-top-social-media-internet-usage.")
    st.write(
        "•   How can we improve PH Democracy? [online] rappler.com. Available at: https://www.rappler.com/nation/philippine-democracy-quality-participation")
    st.write(
        "•   Filipinos Lead the World in... Social Media 'Addiction' [online] esquiremag.ph. Available at: https://www.esquiremag.ph/money/industry/filipinos-social-media-addiction")


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
