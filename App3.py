import streamlit as st
import pickle
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler

data = pd.DataFrame({
    "radius_mean": [0.0],
    "texture_mean": [0.0],
    "perimeter_mean": [0.0],
    "area_mean": [0.0],
    "smoothness_mean": [0.0],
    "compactness_mean": [0.0],
    "concavity_mean": [0.0],
    "concave_points_mean": [0.0],
    "symmetry_mean": [0.0],
    "fractal_dimension_mean": [0.0],
    "radius_se": [0.0],
    "texture_se": [0.0],
    "perimeter_se": [0.0],
    "area_se": [0.0],
    "smoothness_se": [0.0],
    "compactness_se": [0.0],
    "concavity_se": [0.0],
    "concave_points_se": [0.0],
    "symmetry_se": [0.0],
    "fractal_dimension_se": [0.0],
    "radius_worst": [0.0],
    "texture_worst": [0.0],
    "perimeter_worst": [0.0],
    "area_worst": [0.0],
    "smoothness_worst": [0.0],
    "compactness_worst": [0.0],
    "concavity_worst": [0.0],
    "concave_points_worst": [0.0],
    "symmetry_worst": [0.0],
    "fractal_dimension_worst": [0.0],
})


def add_sidebar():
    st.sidebar.header("Cell Nuclei Measurements")
    slider_labels = [
        ("Radius (mean)", "radius_mean"),
        ("Texture (mean)", "texture_mean"),
        ("Perimeter (mean)", "perimeter_mean"),
        ("Area (mean)", "area_mean"),
        ("Smoothness (mean)", "smoothness_mean"),
        ("Compactness (mean)", "compactness_mean"),
        ("Concavity (mean)", "concavity_mean"),
        ("Concave points (mean)", "concave_points_mean"),
        ("Symmetry (mean)", "symmetry_mean"),
        ("Fractal dimension (mean)", "fractal_dimension_mean"),
        ("Radius (se)", "radius_se"),
        ("Texture (se)", "texture_se"),
        ("Perimeter (se)", "perimeter_se"),
        ("Area (se)", "area_se"),
        ("Smoothness (se)", "smoothness_se"),
        ("Compactness (se)", "compactness_se"),
        ("Concavity (se)", "concavity_se"),
        ("Concave points (se)", "concave_points_se"),
        ("Symmetry (se)", "symmetry_se"),
        ("Fractal dimension (se)", "fractal_dimension_se"),
        ("Radius (worst)", "radius_worst"),
        ("Texture (worst)", "texture_worst"),
        ("Perimeter (worst)", "perimeter_worst"),
        ("Area (worst)", "area_worst"),
        ("Smoothness (worst)", "smoothness_worst"),
        ("Compactness (worst)", "compactness_worst"),
        ("Concavity (worst)", "concavity_worst"),
        ("Concave points (worst)", "concave_points_worst"),
        ("Symmetry (worst)", "symmetry_worst"),
        ("Fractal dimension (worst)", "fractal_dimension_worst"),
    ]
    input_dict = {}
    for label, key in slider_labels:
        input_dict[key] = st.sidebar.slider(label, value=0.0)
    return input_dict

def get_scaled_values(input_dict):
    scaled_dict = {}
    for key, value in input_dict.items():
        max_value = data[key].max()
        min_value = data[key].min()

        if max_value == min_value:
            scaled_value = 0.0  # Otra forma de manejar la situación especial
        else:
            scaled_value = (value - min_value) / (max_value - min_value)



        scaled_dict[key] = scaled_value
    return scaled_dict

def get_radar_chart(input_data):
    input_data = get_scaled_values(input_data)
    categories = ["Radius","Texture","Perimeter","Area",
    "Smoothness","Compactness","Concavity", "Concave points",
    "Symmetry","Fractal Dimension"
    ]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
       r=[
          input_data['radius_mean'], input_data['texture_mean'], input_data['perimeter_mean'],
          input_data['area_mean'], input_data['smoothness_mean'], input_data['compactness_mean'],
          input_data['concavity_mean'], input_data['concave_points_mean'], input_data['symmetry_mean'],
          input_data['fractal_dimension_mean']
        ],
        theta=categories,
        fill='toself',
        name='Mean'
    ))
    fig.add_trace(go.Scatterpolar(
         r=[
          input_data['radius_se'], input_data['texture_se'], input_data['perimeter_se'], input_data['area_se'],
          input_data['smoothness_se'], input_data['compactness_se'], input_data['concavity_se'],
          input_data['concave_points_se'], input_data['symmetry_se'],input_data['fractal_dimension_se']
        ],
        theta=categories,
        fill='toself',
        name='Standard Error'
    ))
    fig.add_trace(go.Scatterpolar(
         r=[
          input_data['radius_worst'], input_data['texture_worst'], input_data['perimeter_worst'],
          input_data['area_worst'], input_data['smoothness_worst'], input_data['compactness_worst'],
          input_data['concavity_worst'], input_data['concave_points_worst'], input_data['symmetry_worst'],
          input_data['fractal_dimension_worst']
        ],
        theta=categories,
        fill='toself',
        name='Lowest'
    ))
    fig.update_layout(
    polar=dict(
        radialaxis=dict(
        visible=True,
        range=[0, 1] 
        )),
    showlegend=True
    )

    return fig

def add_prediction(input_data):
    model = pickle.load(open("svm_model4.pkl","rb"))
    scaler = pickle.load(open("scaler4.pkl","rb"))

    input_np = np.array(list(input_data.values())).reshape(1,-1)
    input_scaled = scaler.transform(input_np)

    prediction = model.predict(input_scaled)
    st.subheader("Cell cluster status")
    st.write("The cell cluster is:")

    if prediction[0] == 0:
        st.write("<span class='diagnosis benign'>Benign</span>",unsafe_allow_html = True)
    else:
        st.write("<span class='diagnosis malicious'>Malicious</span>",unsafe_allow_html = True)
    
    st.write("Benign Probability: ", round(model.predict_proba(input_scaled)[0][0],3))
    st.write("Malicious Probability: ", round(model.predict_proba(input_scaled)[0][1],3))
    st.write('The analysis is to purely boost the quality of diagnosis and is not meant as a substitute to professional diagnosis')

def main():
    st.set_page_config(
        page_title = "Breast Cancer Prediction",
        page_icon = ":female-doctor",
        layout = "wide",
        initial_sidebar_state="expanded"
    )



    input_data = add_sidebar()
 

    with st.container():
        st.title("Breast Cancer Predictor")
        st.write("Breast cancer diagnosis often involves the examination of cellular samples obtained through cytology procedures. By integrating our ML application with our cytology lab, we create a comprehensive and efficient workflow that maximizes accuracy and speed in detecting breast cancer.")

    col1, col2 = st.columns([4,1])

    with col1:
        radar_chart = get_radar_chart(input_data)
        st.plotly_chart(radar_chart)
    with col2:
        add_prediction(input_data)

if __name__ == '__main__':
    main() 