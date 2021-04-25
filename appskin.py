import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('modelskin.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('HTMLskin.html')

@app.route('/predict',methods=['POST'])
def predict():

    int_features = [int(x)-1 for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    output = abs(int(prediction[0]))
    res={0:'Eczema Treatment:Lotions and creams to keep skin moist, these should be applied when the skin is damp',1:'Tinea versicolo    Treatment:Fluconazole ,150mg once a week for 4 weeks',2:'Tinea faciei   Treatment:Terbinafine, Itraconazole 4-6 weeks plus topical',3:'Melanoma    Treatment:radiation treatments',4:'vitiligo    Treatment:surgical treatment',5:'acne   Treatment:treated with creams'}
    output=res[output]

    return render_template('HTMLskin.html', prediction_text='Predicted Disease is {}'.format(output))

@app.route('/results',methods=['POST'])
def results():

    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)