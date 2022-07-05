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

    """ #Fingerprint
    f_extended_connectivity = request.form['fec']
    f_atom_environments = request.form['fae']
    f_topological_torsion = request.form['ftts']
    f_atom_pair = request.form['fap']

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

    input='1 80 94 114 237 241 255 294 392 411 425 695 743 747 786 875 1057 1171 1238 1365 1380 1452 1544 1750 1773 1853 1873 1970'

    model_type='smiles'
    result = model_call(input, "TT", model_type)



    #result = model_call(user_input)

    return(render_template('result.html', variable=result))

if __name__ == "__main__":
    app.run(debug=True)