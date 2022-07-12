# main.py
import nltk
from flask import request
from flask import jsonify
from flask import Flask, render_template
from MolForge import model_call

app = Flask(__name__)

@app.route('/')
def my_form():
    return render_template('test.html')

@app.route('/test')
def home():
    return render_template('test.html')

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

    fingerprint, model_type, index, input, class_mol_similarity = '', 'smiles', 'alpha', '',''

    #----------------------------
    #  Fingerprint
    #----------------------------

    if(request.form['fec']):
        fingerprint = "Extended-Connectivity Fingerprint(ECFP)"
        input = request.form['fec']

    elif(request.form['fae']):
        fingerprint = "Atom environments"
        input = request.form['fae']

    elif(request.form['ftts']):
        fingerprint = "Topological torsion-sparse"
        input = request.form['Topological torsion-sparse']

    elif(request.form['fap']):
        fingerprint = "Atom pair-hashed"
        input = request.form['fap']

    #----------------------------
    #  smiles
    #----------------------------

    if(request.form['smiles']):
        model_type = 'smiles'

    elif(request.form['selfies']):
        model_type = 'selfies'

    #----------------------------
    #  tverskiy index parameter
    #----------------------------

    if(request.form['alpha']):
        index = 'alpha'
    elif(request.form['beta']):
        index = 'beta'

    #----------------------------
    #  molecular similarity
    #----------------------------

    #Please modify accordingly -> later
    if(request.form.get('predefined_sub')):
        class_mol_similarity = 'predefined_sub'
    elif(request.form.get('path_feature')):
        class_mol_similarity = 'path_feature'
    elif(request.form.get('path_based')):
        class_mol_similarity = 'path_based'
    elif(request.form.get('four_atom')):
        class_mol_similarity = 'four_atom'
    elif(request.form.get('circular')):
        class_mol_similarity = 'circular'

    # Default input, please update later
    if(len(input) == 0):
        input = '1 80 94 114 237 241 255 294 392 411 425 695 743 747 786 875 1057 1171 1238 1365 1380 1452 1544 1750 1773 1853 1873 1970'
        fingerprint = "Extended-Connectivity Fingerprint(ECFP)"

    result = model_call(input, fingerprint, model_type)


    return(render_template('result.html', variable=result))

if __name__ == "__main__":
    app.run(debug=True)