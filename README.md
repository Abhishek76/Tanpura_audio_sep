# Tanpura_audio_sep

Clone/Download the repo 

Create a virtualenv in python3
https://www.geeksforgeeks.org/python-virtual-environment/

install pip


install dependencies using pip
        Flask 
        scikit-learn 
        librosa 
        pydub 
        matplotlib


pip install Flask scikit-learn librosa pydub matplotlib 

open a terminal from the folder location

and start a simple python server in 0.0.0.0:8000

python server.py 0.0.0.0 port 8000

open another terminal from the folder location

and start flask in the virtual env where you made and installed the dependencies 

I did in a virtualenv named env

source env/bin/activate
set FLASK_ENV=development
set FLASK_APP=./webapp.py
export set FLASK_APP=webapp
echo $FLASK_APP
flask run --host=0.0.0.0




