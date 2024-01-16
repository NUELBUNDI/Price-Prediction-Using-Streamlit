import streamlit as st
import base64
import pandas as pd
from pipeline import *

# Page Configuration
st.set_page_config(page_title="Machine Deploy App", page_icon=":rocket:", layout="wide")

# Header
st.markdown("<h1 style='text-align: center; color: #F3722C;'>Car Price Prediction</h1>", unsafe_allow_html=True)

# Data
@st.cache_data
def get_data(url):
    df = pd.read_csv(url)
    return df

df = get_data('https://raw.githubusercontent.com/NUELBUNDI/Machine-Learning-Data-Set/main/ford.csv')

@st.cache_resource
def load_modelrfr():
    model = joblib.load('decisiontreeregressor_model_3.joblib')
    return model

@st.cache_resource
def load_modelgbr():
    model = joblib.load('gradientboostingregressor_model_21.joblib') 
    return model


# Layout of the Inputs


with st.form("my_form", clear_on_submit=True):
    
    row1 = st.columns([2,2])
    
    year      = row1[0].number_input('Year', value=2015, step=1)
    mileage   = row1[1].number_input(' Input the Car Mileage', value=0)
    
        
    row2 = st.columns([2,2])
    tax        = row2[0].number_input(' Input the Tax', value=0, step=5)
    mpg        = row2[1].number_input('MPG', value=0, step=1)
    
    row3 = st.columns([2,2])
    
    eng_size     = row3[0].slider(' Input the Enginesize', max_value=4.0, min_value=1.0,step=0.5,)
    model        = row3[1].selectbox(' Select Car  Model', options= df['model'].unique().tolist())
        
    row4 = st.columns([2,2])
    
    fuel_type    = row4[0].selectbox(' Select the Fuel Type', options= df['fuelType'].unique().tolist())
    transmission = row4[1].selectbox(' Select the Transmission', options= df['transmission'].unique().tolist())
    
    st.write(" ")
    
    submitted = st.form_submit_button('Predict the Price of the car')
    if submitted:
        dict_= {'year': year,'mileage': mileage, 'tax': tax,'mpg': mpg, 'eng_size':eng_size}
        # st.write(dict_)
        # st.write(year, mileage, tax, mpg, eng_size, model, fuel_type, transmission)
        # st.write(f'{fuel_type_(fuel_type)}')
        # st.write(f'{transmission_(transmission)}')
        # st.write(f'{models(model)}')
        main_dict = {**dict_, **fuel_type_(fuel_type), **transmission_(transmission), **models(model)}
        # st.write(main_dict)
        df = pd.DataFrame([main_dict])
        
        data = df[['year','mileage', 'tax','mpg', 'eng_size',' B-MAX',' C-MAX',' EcoSport',' Edge',' Escort',' Fiesta',
                    ' Focus',' Fusion',' Galaxy',' Grand C-MAX',' Grand Tourneo Connect',' KA',' Ka+',
                    ' Kuga',' Mondeo',' Mustang',' Puma',' Ranger',' S-MAX',' Streetka',' Tourneo Connect',' Tourneo Custom',
                    ' Transit Tourneo','Focus',
                    'Automatic','Manual','Semi-Auto','Diesel','Electric','Hybrid','Other','Petrol']]
        #
        # st.table(data)
        
        # Get the Model
        # model = import_model(options)
        rows7 = st.columns([2,2])
        rows7[0].markdown("#### :rainbow[Decision Tree Regressor Prediction]")
        rows7[1].markdown("#### :rainbow[Gradient Boosting Regressor Prediction]")
        
        
        
        rows6 = st.columns([2,2])
        
        
    
        # Model 1
        
        model_rfr  = load_modelrfr()
        price      = predict_function(data,model_rfr)
        formatted_pric= f"{int(price[0]):,}"
        rows6[0].success(formatted_pric)
        
        # Model 2
        mode_gbr    = load_modelgbr()
        price_two   = predict_function(data,mode_gbr)
        formatted_price = f"{int(price_two[0]):,}"
        rows6[1].success(formatted_price)
        
        

    











##################################### Page info ##############################################################################


github_link = "https://github.com/YourGitHubUsername/YourRepositoryName"
st.sidebar.markdown(f"[GitHub Repository]({github_link})")

st.sidebar.markdown("#### :brown[Contact Me]")

st.sidebar.markdown(
    """<a href="https://github.com/NUELBUNDI/"> 
    <img src="data:image/png;base64,{}" width="25">
    </a>""".format(
        base64.b64encode(open("assets/github.png", "rb").read()).decode()
    ),
    unsafe_allow_html=True,
)
st.sidebar.write(" ")

st.sidebar.markdown(
    """<a href="https://www.linkedin.com/in/lee-emmanuel-24055a116/">
    <img src="data:image/png;base64,{}" width="25">
    </a> """.format(
        base64.b64encode(open("assets/linkedin.png", "rb").read()).decode()
    ),
    unsafe_allow_html=True,
)






