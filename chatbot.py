#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#Myanmar History Education and Trivia Chatbot

#Task a: Expand conversational capabilities to discuss historical events, figures, civilizations, cultural heritage, and significant milestones in human history.
#Task b: Implement rule-based logical reasoning for contextualizing historical facts, cross-referencing timelines, and answering trivia questions with accuracy.
#Task c: Train a CNN model to classify images of historical artifacts, archaeological sites, or ancient ruins for visual exploration and educational enrichment.

# Initialise Wikipedia agent
import wikipedia

#  Initialise AIML agent
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import filedialog
from nltk.sem import Expression, logic
from nltk.inference import ResolutionProver
import speech_recognition as sr
from gtts import gTTS
import pygame
import os
read_expr = Expression.fromstring

# Load the KB
kb = []
data = pd.read_csv('kb.csv', header=None)
[kb.append(read_expr(row)) for row in data[0]]

categories = ["ananda temple", "aung san suu kyi", "bagan", "golden rock", "inle lake", "mandalay hill", "min aung hlaing", "mrauk u", "myanmar flag", "shwedagon pagoda", "u bein bridge"]

# Load the model
model = load_model('image-recognition/model.h5')

import aiml
# Create a Kernel object. No string encoding (all I/O is unicode)
kern = aiml.Kernel()
kern.setTextEncoding(None)
# Use the Kernel's bootstrap() method to initialize the Kernel. The
# optional learnFiles argument is a file (or list of files) to load.
# The optional commands argument is a command (or list of commands)
# to run after the files are loaded.
# The optional brainFile argument specifies a brain file to load.
kern.bootstrap(learnFiles="mybot-basic.xml")

ai_kb = pd.read_csv('AI.csv')
questions = ai_kb['Question'].tolist()
answers = ai_kb['Answer'].tolist()

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import re

#processes text to remove punctuation, convert to lowercase, and remove stopwords
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]','',text)
    text = ' '.join([word for word in text.split() if word not in ENGLISH_STOP_WORDS])
    return text

#preprocesses the questions and fits to tfidf matrix, creating tf/idf vector values for each question
preprocessed_questions = [preprocess_text(question) for question in questions]
vectoriser = TfidfVectorizer()
tfidf_matrix = vectoriser.fit_transform(preprocessed_questions)

#calculates similarity score between user input and questions in Q/A file, returning the most similar question and its associated answer
def similarity(userInput, tfidf_matrix, answers, vectoriser, threshold=0.5):
    userInput = preprocess_text(userInput)
    userInput_vector = vectoriser.transform([userInput])
    cosine_similarities = cosine_similarity(userInput_vector, tfidf_matrix).flatten()

    most_similar_question = np.argmax(cosine_similarities)
    similarity_score = cosine_similarities[most_similar_question]

    if similarity_score < threshold:
        return "Sorry, I do not know that. Be more specific!", similarity_score
    else:
        return answers[most_similar_question], similarity_score

#function to handle user feedback, couldn't get appending to work
def handle_feedback():
    global ai_kb
    feedback = input("Was this answer helpful? (yes/no): ")
    if feedback.lower() == "no":
        addition = input("Would you like to add this question to my knowledge base so I can correct it? (yes/no): ")
        if addition.lower() == "yes":
            new_question = input("Please repeat the question so I can learn from it: ")
            new_answer = input("What is the correct answer? ")
            ai_kb = ai_kb.append({'Question': new_question, 'Answer': new_answer}, ignore_index=True)
        elif addition.lower() == "no":
            print("Ok! Anything else you would like to know?")
    elif feedback.lower() == "yes":
        print("Glad I could help!")

# Checks if a fact is true or false, used foor logical reasoning kb
def check_fact(expr, kb):
    negated_expr = expr.negate()
    if ResolutionProver().prove(expr, kb, verbose=False):
        return "Yes, that is true."
    elif ResolutionProver().prove(negated_expr, kb, verbose=False):
        return "No, that is false."
    else:
        return "Sorry, I don't know."

def process_images(image_path):
    img = image.load_img(image_path, target_size=(256, 256))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

root = tk.Tk()
root.withdraw()

# checks for microphone input using google's speech recognition library
def recognise_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Im listening, ask away!")
        audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio)
            print("You said: ", text)
            return text
        except sr.UnknownValueError:
            print("Sorry, I did not get that.")
            return None
        except sr.RequestError:
            print("Sorry, my speech service has ran into an error. Please try again.")
            return None

# converts the recognised speech to text and stores it as mp3 file to be played back
def speak(text):
    tts = gTTS(text)
    tts.save("voice.mp3")
    pygame.mixer.init()
    pygame.mixer.music.load("voice.mp3")
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)
    pygame.mixer.music.unload()
    os.remove("voice.mp3")

# allows user to choose between typing or speaking to the chatbot
def get_input_mode():
    while True:
        mode = input("Would you like to type or speak? (type/speak): ")
        if mode in ["type", "speak"]:
            return mode
        else:
            print("Invalid input. Please try again.")

# allows user to choose between listening or reading the chatbot's responses
def get_output_mode():
    while True:
        mode = input("Would you like to listen to my responses or read? (listen/read): ")
        if mode in ["listen", "read"]:
            return mode
        else:
            print("Invalid input. Please try again.")

# opens file explorer for user to select an image
def select_img():
    root.deiconify()
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
    root.withdraw()
    return file_path

# filters out stop words and capitalizes the first letter of the input for kb use
def filter_input(input_str):
    words = input_str.lower().split()
    filtered_words = [word for word in words if word not in ENGLISH_STOP_WORDS]
    return ''.join(filtered_words).capitalize()

# Welcome user
print("Hello! I am a chatbot dedicated to exploring Myanmar's rich history. \nI have information regarding festivities, culture, landmarks, and historical time periods of Myanmar, such as: \n Thingyan water festival\n Ethnic groups\n Pyu city-states\n The Bagan Kingdom\n Colonial period under British rule\n The struggle for independence\n Myanmar's struggle for democracy\n \nIf you would like to indentify an image of Myanmar, type 'identify image'!\nIf you would like to try my text-to-speech services, tpye 'tts'!\nIf you have any questions regarding Myanmar then ask away!\n")

current_input_mode = "type"
current_output_mode = "read"

# Main loop
while True:

    # function to output response based on output mode
    def output_response(response):
        if current_output_mode == "read":
            print(response)
        elif current_output_mode == "listen":
            print(response)
            speak(response)

    # get user input
    try:
        if current_input_mode == "speak":
            userInput = recognise_speech()
            if userInput is None:
                continue
        elif current_input_mode == "type":
            userInput = input("> ")
    except (KeyboardInterrupt, EOFError) as e:
        print("Thank you for chatting with me! Goodbye!")
        break

    
    processed_answer, similarity_score = similarity(userInput, tfidf_matrix, answers, vectoriser)
    if similarity_score >= 0.5:
        output_response(processed_answer)
        handle_feedback()
        continue
    #pre-process user input and determine response agent (if needed)
    responseAgent = 'aiml'
    #activate selected response agent
    if responseAgent == 'aiml':
        answer = kern.respond(userInput)
    
    if userInput.lower() == "tts":
        current_input_mode = get_input_mode()
        current_output_mode = get_output_mode()
        if current_input_mode == "speak":
            output_response("You have selected to speak to me. I will now listen to your questions.")
        if current_input_mode == "type":
            output_response("You have selected to type to me.")
        if current_output_mode == "listen":
            output_response("You have selected to listen to my responses.")
        if current_output_mode == "read":
            output_response("You have selected to read my responses.")
        continue

    #post-process the answer for commands
    if answer[0] == '#':
        params = answer[1:].split('$')
        cmd = int(params[0])
        if cmd == 0:
            output_response(params[1])
            break

        elif cmd == 1:
            try:
                wSummary = wikipedia.summary(params[1], sentences=3,auto_suggest=False)
                output_response(wSummary)
                handle_feedback()
            except:
                output_response("Sorry, I do not know that. Be more specific!")
        
        elif cmd == 2:
            img_path = select_img()
            if img_path:
                processed_image = process_images(img_path)
                prediction = model.predict(processed_image)
                predicted_class_index = np.argmax(prediction, axis=1)[0]
                predicted_class_name = categories[predicted_class_index]
                output_response(f"This image is: {predicted_class_name}")
            else:
                output_response("No image selected.")
            continue

        elif cmd == 31: # if input pattern is "I know that * is *"
            object,subject=params[1].split(' is ')
            filtered_obj = filter_input(object)
            filtered_sub = filter_input(subject)
            expr = read_expr(f"{filtered_sub}({filtered_obj})")
            
            if check_fact(expr, kb) == "No, that is false.":
                output_response("Are you sure? This statement contradicts my current knowledge base.")
            elif check_fact(expr, kb) == "Yes, that is true.":
                output_response("Yes, that fact is in my head too!")
            else:
                kb.append(expr)
                output_response(f"OK, I will remember that {filtered_obj} is {filtered_sub}")
            
        elif cmd == 32: # if the input pattern is "check that * is *"
            
            object,subject=params[1].split(' is ')
            filtered_obj = filter_input(object)
            filtered_sub = filter_input(subject)
            expr = read_expr(f"{filtered_sub}({filtered_obj})")

            if check_fact(expr, kb) == "Yes, that is true.":
                output_response("Correct.")
            elif check_fact(expr, kb) == "No, that is false.":
                output_response("Incorrect.")
            else:
                output_response("Sorry, I don't know.")

        elif cmd == 99:
            output_response("I did not get that, please try again.")
    else:
        output_response(answer)