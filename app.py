##############################################################################
#importing the important libraries
##############################################################################

from flask import Flask,render_template,url_for,request
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import load_model

#import pytesseract
#import shutil
#import os
#import random
#try:
# from PIL import Image
#except ImportError:
# import Image
#import pickle

##############################################################################
#pre-requisites variables and other essentials.
##############################################################################
#keras
SEQUENCE_LENGTH = 300

# SENTIMENT
POSITIVE = "POSITIVE"
NEGATIVE = "NEGATIVE"
NEUTRAL = "NEUTRAL"
SENTIMENT_THRESHOLDS = (0.4, 0.7)



##############################################################################
# load the model from disk
##############################################################################
prediction_model = load_model("https://drive.google.com/file/d/1iNzjNMZK9K5lqoAAphjAdumoU9PopGC2/view?usp=sharing")
tokenizer = Tokenizer()
#token_model =pickle.load(open('tokenizer.pkl','rb'))


app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')





def decode_sentiment(score, include_neutral=True):
    if include_neutral:        
        label = NEUTRAL
        if score <= SENTIMENT_THRESHOLDS[0]:
            label = NEGATIVE
        elif score >= SENTIMENT_THRESHOLDS[1]:
            label = POSITIVE

        return label
    else:
        return NEGATIVE if score < 0.5 else POSITIVE

def prediction(text, include_neutral=True):
    #start_at = time.time()
    # Tokenize text
    x_test = pad_sequences(tokenizer.texts_to_sequences([text]), maxlen=SEQUENCE_LENGTH)
    # Predict
    score = prediction_model.predict([x_test])[0]
    # Decode sentiment
    label = decode_sentiment(score, True)

    return label

def predict():
    
    if request.method == 'POST':
        message = request.form['message']
        data = str(message)
        #image_path_in_colab=message
        #extracted_text=pytesseract.image_to_string(Image.open(image_path_in_colab))
        my_label = prediction(data)
    return render_template('result.html',label = my_label)
 
@app.route('/predict',methods=['POST'])
def predictLabel():
    return predict()

if __name__ == '__main__':
	app.run(debug=True)