# main.py
import nltk
from flask import request
from flask import jsonify
from flask import Flask, render_template
from MolForge import model_call

app = Flask(__name__)

@app.route('/')
def my_form():
    return render_template('index.html')

@app.route('/home')
def home():
    return render_template('index.html')

@app.route('/predict')
def predict():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/reference')
def reference():
    return render_template('reference.html')

@app.route('/', methods=['POST'])
def my_form_post():
    user_input = request.form['text']

    #result = model_call(user_input)

    return(render_template('predict.html', variable=user_input))

if __name__ == "__main__":
    app.run(debug=True)