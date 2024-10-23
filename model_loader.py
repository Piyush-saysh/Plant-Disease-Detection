import tensorflow as tf
from keras.layers import TFSMLayer
import requests
from bs4 import BeautifulSoup
from gtts import gTTS
from googletrans import Translator
import pygame

PLANTS = ["model","Apple", "Bell Pepper","Corn (Maize)", "Potato", "Tomato", "Peach","Grape"]

DISEASES = {
    "model":["glioma","meningioma","notumor", "pituitary"],
    "Apple": ["Apple Scab", "Black Rot", "Cedar Black Rust", "Healthy"],
    "Bell Pepper": ["Bacterial Spot", "Healthy"],
    "Corn (Maize)": ["Cercospora Leaf Spot","Common Rust","Healthy","Northern Leaf Blight"],
    "Peach": ["Bacterial Spot", "Healthy"],
    "Potato": ["Early Blight", "Healthy", "Late Blight"],
    "Tomato": ["Bacterial Spot", "Early Blight", "Healthy", "Late Blight", "Septoria Leaf Spot", "Yellow Leaf Curl Virus"],
    "Grape": ["Black rot","Esca (Black Measles)","Healthy","Leaf Blight"]

}

dir_path = "Models"
# dir_path = "/Users/piyushverma/Downloads/Plant-Disease-Predictor-main/Models"


def load_model(plant):
    model_path = f"{dir_path}/{plant}.keras"
    model = tf.keras.models.load_model(filepath=model_path)
    return model


# def load_model(plant):
#     model_path = f"{dir_path}/{plant}.keras"
#     tfsmlayer = TFSMLayer(model_path, call_endpoint='serving_default')
#     model = tf.keras.Sequential([tfsmlayer])
#     return model


def get_plants():
    return PLANTS

# https://www.google.com/search?client=safari&rls=en&q=solution+for+leaf+blight+problem&ie=UTF-8&oe=UTF-8

def get_solution(query):
    translator = Translator()
    url = f"https://www.google.com/search?q=solution+for+{query}+problem"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
    }

    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Find the first paragraph (wrapped in <p> tags) from the search results
    first_paragraph = soup.find('div', class_='BNeawe').text
    language = 'hi'
    translated_test = translator.translate(first_paragraph, dest= language).text
    
    return [first_paragraph,translated_test]

def read_solution(top_snippet, language):

    myobj = gTTS(text=top_snippet, lang=language,slow = False)
    myobj.save("welcome.mp3") 
    pygame.mixer.init()
    pygame.mixer.music.load("welcome.mp3")
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        continue

def get_disease(plant):
    return DISEASES.get(plant)


