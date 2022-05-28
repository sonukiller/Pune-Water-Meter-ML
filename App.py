import numpy as np
import pandas as pd
import datetime

import plotly.express as px
import plotly.graph_objects as go
from streamlit_folium import folium_static
import folium

from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import StandardScaler

import streamlit as st

st.title("Anomaly Detection using Machine Learning")

data = pd.read_csv('Daily water Consumption data NEW.csv')
data['Date'] = pd.to_datetime(data['Date'])

def week_of_month(day):
    if day<28:
        return day//7+1
    else:
        return 4
    
data['Day_of_week'] = data['Date'].dt.dayofweek
data['Week_of_month'] = data['Day'].apply(week_of_month)

# Adding the pipe size to the data
pipe_size = pd.read_csv('Pipe_size.csv', usecols = ['Meter', 'Pipe size'])
data = pd.merge(data, pipe_size, on='Meter')

# PCA
df_pca = data.copy()

# Considering all the features
def single_meter_training(data, contamination=0.01): # contamination is the percent of data that is anomalous
  
    # Numeric to categorical columns
    categorical_cols = ['Month', 'Pipe size', 'Day_of_week', 'Week_of_month']
    data[categorical_cols] = data[categorical_cols].astype('object')
    
    # Removing Date column before training
    data.set_index('Date', inplace=True)
    
    # One hot encoding
    data_new = pd.get_dummies(data)
    
    return data_new

# Calculating the loss
def get_anomaly_scores(df_original, df_restored):
    loss = np.sum((np.array(df_original) - np.array(df_restored)) ** 2, axis=1)
    loss = pd.Series(data=loss, index=df_original.index)
    return loss

# Anomaly with PCA
def Anomaly_with_PCA(data):
    # Preprocessing
    scalar = StandardScaler()
    scaled_data = scalar.fit_transform(data)
    
    #PCA
    n_components = min(int(data.shape[1]*0.75), data.shape[0])
    pca = PCA(n_components=n_components, random_state=99)
    pca_df = pd.DataFrame(pca.fit_transform(scaled_data))
    pca_df.index = data.index
    
    # Inverting the PCA
    restored_df = pd.DataFrame(pca.inverse_transform(pca_df), index=pca_df.index)
    
    # Scores
    scores = get_anomaly_scores(data, restored_df)
    
    # Threshold
    threshold_value = 0.97 # Can be obtimised
    threshold = scores.quantile(threshold_value)
    Anomaly = (scores>threshold)
    
    return Anomaly, scores

output = pd.DataFrame()

# Meter Address Input
meter_list = list(df_pca['Meter'].unique())
option = st.selectbox(
     'Choose the meter address',
     meter_list)
meter = option  #Input
temp = df_pca[df_pca['Meter']==meter]

temp.sort_values(by='Date', inplace=True)

# Date Input
d = st.date_input(
     "Choose the date",
     datetime.date(2021, 12, 15))

date = d # Input
date = pd.to_datetime(date)

#Consumption Input
Consumption = age = st.slider('Consumption (lit)', -5000, 100000, 100)  # Input
temp = temp.append({'Meter':meter, 'Month':date.month, 'Day':date.day, 'Consumption': Consumption, 'Date':date, 'Day_of_week':date.dayofweek, 
            'Week_of_month':week_of_month(date.day), 'Pipe size': list(pipe_size[pipe_size['Meter']==meter]['Pipe size'])[0]}, ignore_index=True)

# Setting up the data
data = temp.drop(columns=['Meter', 'Day'])

# Feature Engineeing, Training and Prediction
one_block = single_meter_training(data)

temp.set_index('Date', inplace=True)
temp['Anomaly'], _ = Anomaly_with_PCA(one_block)
temp = temp[['Meter', 'Consumption', 'Pipe size', 'Anomaly']]
temp.reset_index(inplace=True)

# Appending
output = output.append(temp, ignore_index=True)
output.set_index('Date', inplace=True)

# plot value on y-axis and date on x-axis
fig = px.line(output, x=output.index, y="Consumption", title='Anomaly Detection', template = 'plotly_dark')
# create list of outlier_dates
outlier_dates = output[output['Anomaly'] == True].index
# obtain y value of anomalies to plot
y_values = [output.loc[i]['Consumption'] for i in outlier_dates]

fig.add_trace(go.Scatter(x=outlier_dates, y=y_values, mode = 'markers', 
                name = 'Anomaly', 
                marker=dict(color='red',size=10)))

result = temp.iloc[-1]['Anomaly']

if result == False:
    st.write('The data point is not Anamolus.')
else:
    st.write('The data point is Anamolus!')

    
# fig, ax = plt.subplots()
st.plotly_chart(fig)

# Location
df_map = pd.read_csv('gps.csv')
df_map.rename(columns={'METER_ADDRESS':'Meter'}, inplace=True)

# center on Liberty Bell
latitude = list(df_map[df_map['Meter']==meter]['X Coordinate'])[0]
longitude = list(df_map[df_map['Meter']==meter]['Y Coordinate'])[0]

m = folium.Map(location=[latitude, longitude], zoom_start=16)
# add marker for Liberty Bell
tooltip = meter
folium.Marker(
    [latitude, longitude], popup="Liberty Bell", tooltip=tooltip
).add_to(m)

# call to render Folium map in Streamlit
folium_static(m)
