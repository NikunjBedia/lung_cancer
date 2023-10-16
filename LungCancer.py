from typing import Dict
import streamlit as st
from PIL import Image
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import urllib.request
import ssl
import base64
import json
from io import BytesIO


def number(x):
    if (x==0):
        return 0
    elif (x==1):
        return 1
    elif (x==2):
        return 2
    else:
        return 3


@st.cache_data
def load_image(image_file):
    img = Image.open(image_file)
    # Add preprocessing steps here if needed
    return img



def get_static_store() -> Dict:
    """This dictionary is initialized once and can be used to store the files uploaded"""
    return {}

def color_low_confidence(val):
    """
    Takes a scalar and returns a string with
    the css property `'background-color: red'` for
    values that match 'low confidence', and an
    empty string otherwise.
    """
    color = 'red' if val == 'low confidence' else ''
    return f'background-color: {color}'

def highlight_cells(val):
    return f"height: {100}px"
pic_array = []
def predict(result):
    pred = []
    cls = []
    rmrk = []
    image_name = []
    prob1 = []
    prob2 = []
    prob3 = []
    prob4 = []
    button=[]
    count = 0  # Used in progress bar calculation
    corr = 0  # corrupt files count
    progress_text = "Image(s) Processing..."
    prog_bar = st.progress(count, text=progress_text)

    
    image_data_list = []    

    if result:
        # Loop through the uploaded files and make predictions
        for uploaded_file in result:
            count+=1
            # Load the image from the file
            value = Image.open(uploaded_file)
            try:
                with Image.open(uploaded_file) as img:
                    img.verify()
            except(IOError, SyntaxError):
                corr+=1
                continue


            buffered = BytesIO()
            value.save(buffered, format="PNG")
            
            image_data = {
                    'filename': uploaded_file.name,
                    'data': base64.b64encode(buffered.getvalue()).decode('utf-8')
                }
            image_data_list.append(image_data)

        # Convert the list of image data to a JSON object
        body = str.encode(json.dumps({'images': image_data_list}))
                     
            
           
        # Display the predictions in a table
    
    url = 'https://image-analytics-olwpj.westus2.inference.ml.azure.com/score'
    # Replace this with the primary/secondary key or AMLToken for the endpoint
    api_key = 'okJi7vdeAIOo7meJeqkNMLUBZOtGvaVD'
    if not api_key:
        raise Exception("A key should be provided to invoke the endpoint")

    # The azureml-model-deployment header will force the request to go to a specific deployment.
    # Remove this header to have the request observe the endpoint traffic rules
    headers = {'Content-Type':'application/json', 'Authorization':('Bearer '+ api_key), 'azureml-model-deployment': 'resnet50-cancer-3' }

    req = urllib.request.Request(url, body, headers)

    try:
        response = urllib.request.urlopen(req)

        result = response.read()
        # print(result)
        # print(type(result))
        data = json.loads(json.loads(result.decode("utf-8")))
        # print(type(data))

        df = pd.DataFrame(data, columns=["Patient Id", "image_filename", "Actual_class", "predicted_class","Adeno","Largecell","Squamouscell","Normal"])
    except urllib.error.HTTPError as error:
        print("The request failed with status code: " + str(error.code))
                

            
    prog_bar.empty()  # removes progress bar after loading
    
    # df['Adeno']=df['A'].round(3).apply(lambda x: '{:.3f}'.format(x))
    # df['Largecell'] = df['E'].round(3).apply(lambda x: '{:.3f}'.format(x))
    # df['Normal'] = df['N'].round(3).apply(lambda x: '{:.3f}'.format(x))
    # df['Squamouscell'] = df['G'].round(3).apply(lambda x: '{:.3f}'.format(x))

    # df=df.drop(['A', 'E','G','N'], axis=1)



    df_count = df['predicted_class'].value_counts()  # gets freq of each class
    col = []  # dividing columns for each class card
    col = st.columns(len(df_count) + 1)

    colno = 0  # to keep index of column
    for val, c in df_count.items():
        col[colno].metric(label=val, value=c)
        colno += 1
    col[colno].metric(":blue[Total images]", value=count, help="Corrupt images = " + str(corr))
    st.dataframe(df.style.background_gradient(cmap="Reds",
                                              subset=['Adeno', 'Largecell',
                                                      'Normal',
                                                      'Squamouscell'], axis=1,
                                              low=0.5,

                                              high=1),use_container_width=True)




  
    def convert_df(df):
       return df.to_csv(index=False).encode('utf-8')
    
    
    csv = convert_df(df)
    
    st.download_button(
       "Download as CSV",
       csv,
       "file.csv",
       "text/csv",
       key='download-csv'
    )
    # Define the height of each row in pixels

    # Use st.beta_columns to create two columns
    # Set the height of the rows in the left column using CSS styling



def first():

    st.markdown("<h1 style='text-align: center; color: green;'>Lung Cancer Classification</h1>", unsafe_allow_html=True)
    img = Image.open("C:/Users/nikunj.bedia/testing_streamlit/DSA_Repo-main/lung_cancer.jpg")
    resized_img = img.resize((350, 300))
    with st.columns(3)[1]:
        st.image(resized_img)
    st.write("""
    - This Decision Support System serves as an accurate lung cancer classification system that can differentiate between normal and cancerous PET-CT scan images. 
    - The system can classify cancerous images into three distinct types of non-small cell lung cancer: adenocarcinoma, large cell carcinoma, and squamous cell carcinoma. 
    - The objective is to aid in early and accurate diagnosis, allowing for timely treatment planning and improved patient outcomes.
    """)


    result = st.file_uploader("Upload one or more images.", type=["PNG", "JPEG", "JPG"], key="real",
                              accept_multiple_files=True)

    if st.button('Submit'):
        predict(result)
    
    uploaded_images = []
    button_states = []

    # Add a file uploader widget to allow the user to upload images

    # If an image is uploaded, load and preprocess it, and add it to the list of uploaded images and button states

first()









