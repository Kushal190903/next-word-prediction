import streamlit as sl
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np


model = load_model(r"sherlock_model.h5")


with open(r"tokenizer.pkl", "rb") as file:
    tokenizer = pickle.load(file)

# Streamlit app title and description
sl.title("Next Word Prediction App")
sl.markdown("""
    **Welcome to the Next Word Prediction App!**
    
    
""")

# Sidebar information
sl.sidebar.title("About")
sl.sidebar.info("""
    This app uses a pre-trained LSTM based language model to predict the next word in a sequence. 
    The model has been trained on the texts of the popular series 'Sherlock Holmes' by the author 'Sir Arthur Conan Doyle', 
    so predictions are based on patterns learned from that data.
    
    **Instructions:**
    1. Start typing in the text box.
    2. Press enter to trigger prediction.
    3. The predicted word will be displayed below the input box.
                
   **Note:
    This model is designed to handle sequences of length upto 30 words. Though sequences of larger length are allowed, the results may not be as effective.
""")

def predict_next_word(input_text):
    # Tokenize and pad the input text
    input_tokens = tokenizer.texts_to_sequences([input_text])[0]
    input_tokens = pad_sequences([input_tokens], maxlen=30)
    
    # Predict the next word
    prediction = model.predict(input_tokens)
    predicted_word = tokenizer.index_word[np.argmax(prediction)]
    
    return predicted_word


input_text = sl.text_input(label="Start typing your sentence here...")


if input_text :
    predicted_word = predict_next_word(input_text)  
    
    # Display the predicted word
    sl.markdown(f"""
        **Predicted Next Word:** 
        ```{predicted_word}```
    """)
else:
    sl.markdown("**Predicted Next Word:** _(waiting for input)_")

# footer
sl.markdown("""
    ---
    **Note:** This app is the demonstration of a word prediction model. The accuracy of predictions may vary based on the input text.
""")


    

