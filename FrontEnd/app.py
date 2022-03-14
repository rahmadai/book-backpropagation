from types import MethodType
from flask import *
from predict_inference import Predict
import os,csv,json



app = Flask(__name__)


@app.route('/',methods =['GET'])
def index():
    if request.method == 'GET':
        return render_template('index.html')

@app.route('/pengujian',methods =['GET'])
def pengujian():
    if request.method == 'GET':
        return render_template('pengujian.html')

@app.route('/dataset',methods =['GET'])
def dataset():
    if request.method == 'GET':
        return render_template('dataset.html')

@app.route('/accuracy',methods =['GET'])
def chart():
    if request.method == 'GET':
        return render_template('Acc.html')

@app.route('/loss',methods =['GET'])
def loss():
    if request.method == 'GET':
        return render_template('Loss.html')

@app.route('/compare',methods =['GET'])
def compare():
    if request.method == 'GET':
        return render_template('Compare.html')

@app.route('/predict',methods=['POST'])
def search():
    if request.method == 'POST':
        judul = request.json
        result = Predict(judul['text'])
        
        return json.dumps(result)

@app.route('/data')
def data():
    csvfile = open('dataset/set.csv', 'r')

    jsonArray = []
    fieldnames = ("no","judul","penulis","genre","tahun","kategori")
    reader = csv.DictReader( csvfile, fieldnames)
    for row in reader:
        # json.dump(row, jsonfile)
        # jsonfile.write('\n')
        jsonArray.append(row)
    js = {
        "data" : jsonArray
    }
    return json.dumps(js)


if __name__ == '__main__':
    app.run(port=5000,debug=True)
    

