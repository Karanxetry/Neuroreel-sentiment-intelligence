import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

model=load_model('simpleRNN.h5')
word_index=imdb.get_word_index()
final_output=[]
def preprocessing(review):
   
    review = review.lower().split()
    for word in review:
        if word in word_index:
            num_value=word_index.get(word)
            final_output.append(num_value)
        else:
             final_output.append(2)
    
    processed_review=sequence.pad_sequences([final_output],maxlen=100)
    return processed_review

st.title('MOVIE SENTIMENT CLASSIFICATION')
st.write('Enter your review for the movie....')
user_input=st.text_area('Movie Review')

if st.button('Classify'):
    result=preprocessing(user_input)
    output=model.predict(result)
    sentiment='Positive' if output[0][0]>0.5 else 'Negative'

    st.write('Sentiment:', sentiment)
    st.write('Prediction Score is:', output[0][0])
else:
    print('Please enter your movie review')