import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import plotly.express as px
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
import joblib
from tensorflow.keras.models import load_model
@st.cache_data
def load_data():
    data = pd.read_csv('electricity_cost_dataset.csv')
    return data
df = load_data()

st.title('Electricity Consumption Prediction')
c1 = st.sidebar.checkbox('Show data')
if c1:
    st.subheader('Full dataset')
    st.dataframe(df)
c2 = st.sidebar.checkbox('EDA')
if c2:
    st.subheader('Top 5 records')
    st.write(df.head(5))

    st.subheader('Last 5 records')
    st.write(df.tail(5))

    st.subheader('Statistics of data')
    st.write(df.describe())

    st.subheader('Null count')
    st.write(df.isnull().sum())

    st.subheader('Correlation')
    st.write(df.corr(numeric_only=True))

    fig = px.histogram(
        df,
        x='site area',
        nbins=20,
        title='Distribution of Site Area',
        color_discrete_sequence=['#636EFA'],
        opacity=0.85
    )

    fig.update_layout(
        xaxis_title='Site Area',
        yaxis_title='Count',
        template='plotly_white',
        title_x=0.3
    )

    fig.update_traces(
        marker_line_width=1,
        marker_line_color='black'
    )

    st.plotly_chart(fig, use_container_width=True)

    structure_count = df['structure type'].value_counts().reset_index()

    structure_count.columns = ['Structure Type', 'Count']

    fig = px.bar(
        structure_count,
        x='Structure Type',
        y='Count',
        title='Count of Different Structure Types',
        text='Count',
        color='Structure Type'
    )

    fig.update_layout(
        template='plotly_white',
        title_x=0.25,
        xaxis_title='Structure Type',
        yaxis_title='Number of Buildings',
        showlegend=False
    )

    fig.update_traces(
        textposition='outside'
    )

    st.plotly_chart(fig, use_container_width=True)

    cost = df.groupby('structure type')['electricity cost'].mean().reset_index()
    fig = px.bar(
        cost,
        x='structure type',
        y='electricity cost',
        title='Average Electricity Cost',
        text_auto='.2f',
        color='electricity cost',
    )

    fig.update_layout(
        template='plotly_white',
        title_x=0.25,
        xaxis_title='Structure Type',
        yaxis_title='Number of Buildings',
        showlegend=False
    )

    fig.update_traces(
        textposition='outside'
    )

    st.plotly_chart(fig, use_container_width=True)

    fig = px.histogram(
        df,
        x='water consumption',
        nbins=10,
        title='Distribution of water consumption',
        color_discrete_sequence=['#636EFA'],
        opacity=0.85
    )

    fig.update_layout(
        xaxis_title='water consumption',
        yaxis_title='Count',
        template='plotly_white',
        title_x=0.3
    )

    fig.update_traces(
        marker_line_width=1,
        marker_line_color='black'
    )

    st.plotly_chart(fig, use_container_width=True)

    fig = px.histogram(
        df,
        x='resident count',
        nbins=20,
        title='Distribution of resident count',
        color_discrete_sequence=['#636EFA'],
        opacity=0.85
    )

    fig.update_layout(
        xaxis_title='resident count',
        yaxis_title='Count',
        template='plotly_white',
        title_x=0.3
    )

    fig.update_traces(
        marker_line_width=1,
        marker_line_color='black'
    )

    st.plotly_chart(fig, use_container_width=True)

    fig = px.histogram(
        df,
        x='air qality index',
        nbins=20,
        title='Distribution of air qality index',
        color_discrete_sequence=['#636EFA'],
        opacity=0.85
    )

    fig.update_layout(
        xaxis_title='air qality index',
        yaxis_title='Count',
        template='plotly_white',
        title_x=0.3
    )

    fig.update_traces(
        marker_line_width=1,
        marker_line_color='black'
    )

    st.plotly_chart(fig, use_container_width=True)

    fig = px.histogram(
        df,
        x='recycling rate',
        nbins=8,
        title='Distribution of recycling rate',
        color_discrete_sequence=['#636EFA'],
        opacity=0.85
    )

    fig.update_layout(
        xaxis_title='recycling rate',
        yaxis_title='Count',
        template='plotly_white',
        title_x=0.3
    )

    fig.update_traces(
        marker_line_width=1,
        marker_line_color='black'
    )

    st.plotly_chart(fig, use_container_width=True)

    index = df.groupby('structure type')['air qality index'].mean().reset_index()
    fig = px.bar(
        index,
        x='structure type',
        y='air qality index',
        title='Average air qality index',
        text_auto='.2f',
        color='air qality index',
    )

    fig.update_layout(
        template='plotly_white',
        title_x=0.25,
        xaxis_title='Structure Type',
        yaxis_title='air qality index',
        showlegend=False
    )

    fig.update_traces(
        textposition='outside'
    )

    st.plotly_chart(fig, use_container_width=True)

    fig = px.box(
        df,
        y='electricity cost',
        title='Electricity Cost Distribution'
    )

    st.plotly_chart(fig, use_container_width=True)

    fig = px.box(
        df,
        y='air qality index',
        title='air qality index Distribution'
    )

    st.plotly_chart(fig, use_container_width=True)

    fig = px.box(
        df,
        y='issue reolution time',
        title='issue reolution time Distribution(Hourly)'
    )

    st.plotly_chart(fig, use_container_width=True)

encode  = LabelEncoder()
df['structure type'] = encode.fit_transform(df['structure type'])

t = st.sidebar.slider("test size",.1,.5)
x = df.drop('electricity cost',axis=1)
y = df['electricity cost']

xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=t,random_state=42,shuffle=True)

xscaler = StandardScaler()
xtrain = xscaler.fit_transform(xtrain)
xtest = xscaler.transform(xtest)
yscaler = StandardScaler()
ytrain = yscaler.fit_transform(ytrain.values.reshape(-1, 1)).ravel()
ytest = yscaler.transform(ytest.values.reshape(-1, 1)).ravel()

shape = st.sidebar.checkbox("train test data shape")
if shape:
    st.write("xtrain shape:",xtrain.shape)
    st.write("ytrain shape:",ytrain.shape)
    st.write("xtest shape:",xtest.shape)
    st.write("ytest shape:",ytest.shape)


if st.sidebar.button("Train Model"):

    model = Sequential([
        Dense(128, activation='relu', input_shape=(xtrain.shape[1],)),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1)
    ])

    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    history = model.fit(
        xtrain,
        ytrain,
        epochs=100,
        batch_size=8,
        validation_split=0.2,
        callbacks=[early_stop]
    )

    loss, mae = model.evaluate(xtest, ytest)


    st.write("MAE:", mae)
    # Save ANN model
    model.save("electricity_model.keras")
    st.success("Model saved successfully")

    # Save scalers
    joblib.dump(xscaler, "xscaler.pkl")
    joblib.dump(yscaler, "yscaler.pkl")

    st.success("Scalers saved successfully")
    loss_df = pd.DataFrame({
        'Epoch': range(1, len(history.history['loss']) + 1),
        'Training Loss': history.history['loss'],
        'Validation Loss': history.history['val_loss']
    })

    fig = px.line(
        loss_df,
        x='Epoch',
        y=['Training Loss', 'Validation Loss'],
        title='Training vs Validation Loss'
    )

    st.plotly_chart(fig, use_container_width=True)

    mae_df = pd.DataFrame({
        'Epoch': range(1, len(history.history['mae']) + 1),
        'Training MAE': history.history['mae'],
        'Validation MAE': history.history['val_mae']
    })

    fig = px.line(
        mae_df,
        x='Epoch',
        y=['Training MAE', 'Validation MAE'],
        title='Training vs Validation MAE'
    )

    st.plotly_chart(fig, use_container_width=True)

if st.sidebar.checkbox("Prediction"):
    st.subheader("Prediction Electricity Cost in $")
    col1, col2 = st.columns(2)
    site_area = col1.number_input('Site Area(sq meter)',400,6000)
    structure_type = col2.selectbox('Structure Type',['Mixed-use','Residential','Commercial','Industrial'])
    water_consumption = col1.number_input('Water Consumption (liters)',800,12000)
    recycling_rate = col2.number_input('Recycling Rate (%)',0,100)
    utilisation_rate = col1.number_input('Utilisation Rate (%)',0,100)
    air_quality_index = col2.number_input('Air Quality Index (%)',0,450)
    issue_resolution_time = col1.number_input('Issue Resolution Time (Hours)',0,85)
    resident_count = col2.number_input('Resident Count',0,700)

    if st.button("Predict cost"):
        model = load_model("electricity_model.keras")
        structure_encoded = encode.transform(
            [structure_type]
        )[0]

        input_data = np.array([[
            site_area,
            structure_encoded,
            water_consumption,
            recycling_rate,
            utilisation_rate,
            air_quality_index,
            issue_resolution_time,
            resident_count
             ]])

        input_scaled = xscaler.transform(input_data)

        prediction = model.predict(input_scaled)

        prediction = yscaler.inverse_transform(prediction)

        st.success(
            f"Predicted Electricity Cost: ${prediction[0][0]:,.2f}"
        )