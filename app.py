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

@app.route('/result', methods=['POST'])
def my_form_post():

    """


    #Output type
    smiles = request.form['smiles']
    selfies = request.form['selfies']

    #Tverskiy index parameter
    alpha = request.form['alpha']
    beta = request.form['beta']

    #Fingerprint class for molecular similarity
    predefined_sub = request.form['predefined_sub']
    path_feature = request.form['path_feature']
    path_based = request.form['path_based']
    four_atom = request.form['four_atom']
    circular = request.form['circular']

    #Fingerprint type for molecular similarity
    atom_pair_hashed = request.form['predefined_sub']
    atom_pair_hashed = request.form['rdk_with']
    atom_pair_hashed = request.form['rdk_without']

    #others
    macc = request.form['macc']
    avalon = request.form['avalon']
    atom_hashed_pair = request.form['atom_hashed_pair'] """

    fingerprint, f_output = "", ""
    #Fingerprint
    if(request.form['fec']):
        fingerprint = 'ECFP4'

    elif(request.form['fae']):
        fingerprint = 'AEs'

    elif(request.form['ftts']):
        fingerprint = 'TT'

    elif(request.form['fap']):
        fingerprint = 'HashAP'

    #smiles
    if(request.form['smiles']):
        f_output = 'smiles'

    elif(request.form['selfies']):
        f_output = 'selfies'


    #result = model_call(input)


    result = "Argen is testing " + fingerprint + f_output

    return(render_template('result.html', variable=result))

if __name__ == "__main__":
    app.run(debug=True)