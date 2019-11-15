# To run, open a virtual python environment and then
# export/set FLASK_ENV=development
# export/set FLASK_APP=./webapp.py
# flask run
# Note to self, file cannot be called app.py because there's a name collision

import os #, requests
from flask import Flask, request, render_template, make_response, jsonify, Response
from werkzeug import secure_filename

from mltools import mfcc_calc
from mltools import clf_SVM
import numpy

import ppo
import pickle
import librosa





f = open('./vocW.pckl', 'rb')
vocW = pickle.load(f)
f.close()

f = open('./noiseW.pckl', 'rb')
noiseW = pickle.load(f)
f.close()

mix , fs = librosa.load("./predict/18_mix.wav")

vocal , noise = ppo.Test_Nmf(mix, vocW , noiseW )

librosa.output.write_wav('./vocal/18vocal.wav', vocal, fs)




