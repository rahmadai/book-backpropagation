from types import MethodType
from flask import *
from predict_inference import Predict
import os



app = Flask(__name__)


@app.route('/',methods =['GET'])
def index():
    if request.method == 'GET':
        return render_template('index.html')

@app.route('/pengujian',methods =['GET'])
def pengujian():
    if request.method == 'GET':
        return render_template('pengujian.html')

@app.route('/chart',methods =['GET'])
def chart():
    if request.method == 'GET':
        return render_template('frame.html')

@app.route('/predict',methods=['POST'])
def search():
    if request.method == 'POST':
        judul = request.json
        result = Predict(judul['text'])
        
        return json.dumps(result)
    
    


if __name__ == '__main__':
    app.run(port=5000,debug=True)
    
    

