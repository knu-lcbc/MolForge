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
    #Fingerprint type for molecular similarity
    atom_pair_hashed = request.form['predefined_sub']
    atom_pair_hashed = request.form['rdk_with']
    atom_pair_hashed = request.form['rdk_without']

    #others
    macc = request.form['macc']
    avalon = request.form['avalon']
    atom_hashed_pair = request.form['atom_hashed_pair'] """

    fingerprint, f_output, index, class_mol_similarity = '', '', '', ''

    #----------------------------
    #  Fingerprint
    #----------------------------

    if(request.form['fec']):
        fingerprint = 'Extended-Connectivity Fingerprint(ECFP)'

    elif(request.form['fae']):
        fingerprint = 'Atom environments'

    elif(request.form['Topological torsion-sparse']):
        fingerprint = 'TT'

    elif(request.form['fap']):
        fingerprint = 'Atom pair-hashed'

    #----------------------------
    #  smiles
    #----------------------------

    if(request.form['smiles']):
        f_output = 'smiles'

    elif(request.form['selfies']):
        f_output = 'selfies'

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

    input = '1 80 94 114 237 241 255 294 392 411 425 695 743 747 786 875 1057 1171 1238 1365 1380 1452 1544 1750 1773 1853 1873 1970'
    result = model_call(input,fingerprint, f_output)


    #result = "Selected values are: " + fingerprint + ' '+ f_output + '  '+  index + '  '+  class_mol_similarity

    return(render_template('result.html', variable=result))

if __name__ == "__main__":
    app.run(debug=True)