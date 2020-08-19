from __future__ import unicode_literals# json for converting string dictionary from logout to dictionary
from django.shortcuts import render
import io
import matplotlib.pyplot as plt
import numpy as np
import urllib,base64
from urllib.parse import urlparse
#ALL THE MODEL LIBRARIES
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import pickle
from tensorflow.keras.models import load_model


def Give_url(figure):
    buf = io.BytesIO()
    figure.savefig(buf, format='png')
    buf.seek(0)
    string = base64.b64encode(buf.read())
    uri = urllib.parse.quote(string)
    return uri


missing_values = ["DATA EXPIRED", "CONSENT REVOKED"]
df = pd.read_csv("static/models/hell.csv", na_values=missing_values)  # OUR DATASET
df.drop(df.columns[0], axis=1, inplace=True)
df.dropna(axis=0, inplace=True)
# IMPORT THE MODEL
single = load_model("static/models/first_model_feeling1longonly_NEWMAIN.hdf5")
MULTI_model = load_model("static/models/multi_traget_feeling.hdf5")
##############################################################################################
# TOKENIZOR OBJECT IMPORTED FROM FILE TO CONVERT STRING INPUT TO INTEGER TOKENS AS USED
# WHILE TRAINING GIVE PATH OF (token_NEWMAIN.pickle) FILE BELOW
with open('static/models/token_NEWMAIN.pickle', 'rb') as handle:
    tokenizortest = pickle.load(handle)

input = []
for i in df["text_long"]:
    input.append(tokenizortest.texts_to_sequences([i])[0])
TOKENIZED_INPUT = np.array(pad_sequences(input, maxlen=561, padding='pre'))

# BELOW CODE TO MAKE DICTIONARY TO INVERSE TRANSFORM PREDICTED RESULT
emotion_output = single.predict(TOKENIZED_INPUT)
unique_emotion, ids = np.unique(np.array(df["chosen_emotion"]), return_inverse=True)
unique_emotion = unique_emotion[emotion_output.argmax(1)]

unique, counts = np.unique(unique_emotion, return_counts=True)
# DIFFERENT EMOTIONS OF WHOLE ANALYSIS
plt.figure()
plt.bar(unique, counts)
plt.grid()
fig = plt.gcf()
Diff_emotions=Give_url(fig)

#Pie chart plot
plt.figure()
plt.axis('equal')
plt.pie(counts, labels=unique, autopct='%1.2f%%')
fig = plt.gcf()
Piechart = Give_url(fig)

# HAPPINESS INDEX OF ALL COUNTRIES
data = df[["Nationality", "happiness"]]

data = data.groupby(['Nationality']).mean().reset_index()
data["happiness"] = data["happiness"].round(1)
nationality = np.array(data["Nationality"])
haapy_index = np.array(data["happiness"])
plt.figure(figsize=(15, 20))
plt.barh(nationality, haapy_index, alpha=0.5, color='green')
plt.title("Happiness Index Of Countries", fontsize=30)
plt.yticks(nationality, fontsize=18)
plt.xticks(np.arange(0, 10, 1), fontsize=20)
plt.xlabel("Happiness Index", fontsize=27)
plt.grid()
for index, value in enumerate(haapy_index):
    plt.text(value, index, str(value), fontsize=17)
fig = plt.gcf()
Happiness_index = Give_url(fig)

# CONVERT EACH EMOTION COLUMN TO BINARY CATEGORICAL VALUE
uniques_worry, ids_worry = np.unique(np.array(df["worry"]), return_inverse=True)

uniques_anger, ids_anger = np.unique(np.array(df["anger"]), return_inverse=True)

uniques_fear, ids_fear = np.unique(np.array(df["fear"]), return_inverse=True)

uniques_disgust, ids_disgust = np.unique(np.array(df["disgust"]), return_inverse=True)

uniques_anxiety, ids_anxiety = np.unique(np.array(df["anxiety"]), return_inverse=True)

uniques_sadness, ids_sadness = np.unique(np.array(df["sadness"]), return_inverse=True)

uniques_happiness, ids_happiness = np.unique(np.array(df["happiness"]), return_inverse=True)

uniques_relaxation, ids_relaxation = np.unique(np.array(df["relaxation"]), return_inverse=True)

uniques_desire, ids_desire = np.unique(np.array(df["desire"]), return_inverse=True)


#from celery.schedules import crontab
#from celery.task import periodic_task

#@periodic_task(run_every=crontab(hour=20,minute=45,day_of_month=26))
#def every_monday_morning():
#    print("This is run every Monday morning at 7:30")
from periodically.decorators import *

@hourly()
def my_task():
    print ('Do something!')

@every(minutes=1)
def my_other_task():
    print ('Do something else every 45 minutes!')

def Index(request):
    if(request.method=='POST'):
        MAIN_INPUT=request.POST.get('Input_tweet2')
        print("MAIN_INPUT:"+MAIN_INPUT)
        print("INPUT FROM THE PAGE TWEET HAS BEEN")
    else:
        MAIN_INPUT = "tweet whoes emotion we have to find happy so much"
    input = []
    input.append(tokenizortest.texts_to_sequences([MAIN_INPUT])[0])
    TOKENIZED_INPUT = np.array(pad_sequences(input, maxlen=561, padding='pre'))

    # BELOW CODE TO MAKE DICTIONARY TO INVERSE TRANSFORM PREDICTED RESULT
    uniques, ids = np.unique(df["chosen_emotion"], return_inverse=True)

    # PERFORM PREDICTION AND TRANSFORM RESULT to EMOTION STRING
    PREDICT_EMOTION = single.predict(TOKENIZED_INPUT)
    # BELOW LINE TRANSFORM INTEGER OUTPUT TO EMOTION
    OUT = uniques[PREDICT_EMOTION.argmax(1)]
    OUT=OUT[0]
    print("INPUT:" + str(tokenizortest.sequences_to_texts(TOKENIZED_INPUT)) + " OUTPUT" + str(OUT))

    ##########################################################################
    # THE OUTPUT WILL BE IN BELOW ORDER
    # [worry,anger,disgust,fear,anxiety,sadness,happiness,relaxation,desire]
    ##########################################################################
    worry, anger, disgust, fear, anxiety, sadness, happiness, relaxation, desire = MULTI_model.predict(TOKENIZED_INPUT)
    worry = uniques_worry[worry.argmax(1)]
    anger = uniques_anger[anger.argmax(1)]
    disgust = uniques_disgust[disgust.argmax(1)]
    fear = uniques_fear[fear.argmax(1)]
    anxiety = uniques_anxiety[anxiety.argmax(1)]
    sadness = uniques_sadness[sadness.argmax(1)]
    happiness = uniques_happiness[happiness.argmax(1)]
    relaxation = uniques_relaxation[relaxation.argmax(1)]
    desire = uniques_desire[desire.argmax(1)]

    plt.figure(figsize=(15, 15))
    Feeling = ["worry", "anger", "disgust", "fear", "anxiety", "sadness", "happiness", "relaxation", "desire"]
    Feeling_index = [worry[0], anger[0], disgust[0], fear[0], anxiety[0], sadness[0], happiness[0], relaxation[0],
                     desire[0]]
    plt.bar(Feeling, Feeling_index, alpha=0.5, color='grey')
    plt.title("Feeling Index Of Tweet", fontsize=23)
    plt.xticks(Feeling, fontsize=18)
    plt.yticks(np.arange(0, 11, 1), fontsize=15)
    plt.xlabel("Feelings ", fontsize=20)
    plt.ylabel("Scale", fontsize=20)
    plt.grid()
    for index, value in enumerate(Feeling_index):
        plt.text(index, value, str(value), fontsize=17)
    fig = plt.gcf()
    Tweet_Sentiment = Give_url(fig)
    return render(request, "TrialAnaly.html", {"Diff_emotions": Diff_emotions, "Piechart":Piechart, "Happiness_index":Happiness_index,
                                             "Tweet_Sentiment":Tweet_Sentiment,"Output_Tweet":OUT})


a=0
test_pic=Give_url(fig)

def call_files():
    global a
    print("in call files"+str(a)+"::")
    read_file="static/datasets/sample"+str(a)+".txt"
    pulldata=open(read_file,"r").read()

    if(a+1==10):
        a=0
    else:
        a=a+1
    return pulldata

def test2(request):
    global test_pic
    fig = plt.figure()
    ax12 = fig.add_subplot(1, 1, 1)
    pulled = call_files()
    plt.title("Doing stuff...")
    dataArray = pulled.split('\n')
    xar = []
    yar = []
    for eachline in dataArray:
        if len(eachline) > 1:
            x, y = eachline.split(',')
            xar.append(int(x))
            yar.append(int(y))
    ax12.clear()
    ax12.plot(xar, yar)
    print(xar)

    test_pic = plt.gcf()
    test_pic = Give_url(fig)
    return render(request,"new.html",{"test_pic":test_pic})

from apscheduler.schedulers.blocking import BlockingScheduler


'''
scheduler = BlockingScheduler()
scheduler.add_job(some_job, 'interval',seconds=5)
scheduler.start()
'''



