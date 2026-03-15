import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.express as px

st.title("Tourist Location Recommender")

data = pd.read_csv("tourism.csv")

features = data[['Rating','Cost','Popularity','Adventure','Nature','Culture']]

kmeans = KMeans(n_clusters=3)
data['Cluster'] = kmeans.fit_predict(features)

pca = PCA(n_components=2)
reduced = pca.fit_transform(features)

data['PCA1'] = reduced[:,0]
data['PCA2'] = reduced[:,1]

fig = px.scatter(
    data,
    x="PCA1",
    y="PCA2",
    color="Cluster",
    hover_data=['Destination']
)

st.plotly_chart(fig)

place = st.selectbox("Select Destination", data['Destination'])

cluster = data[data['Destination']==place]['Cluster'].values[0]

recommend = data[data['Cluster']==cluster]['Destination']

st.write("Recommended places:")

for i in recommend:
    if i != place:
        st.write(i)
