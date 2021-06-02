import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from flask import Markup
app = Flask(__name__)
model = pickle.load(open('modelskin.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('HTMLskin.html')

@app.route('/predict',methods=['POST'])
def predict():
    res1={'Itching':1.0,'Skin becomes dry, thickened and scaly':2.0,'Rash-commonly affect face, neck, back of knees and arms':3.0,'Fine, branny scaling':4.0,'Upper trunk, upper arms, neck and abdomen':5.0,"change of skin color, white patches":6.0,'skin peeling':7.0,'Vesicular, pustula':8.0,'Dry type infection':9.0,'moles have asymmetrical shapes':10.0,'uneven colors of moles':11.0,'change in size of moles':12.0,'loss of hair color':13.0,'patches of skin':14.0,'pigmentation in the skin':15.0,'oval macules':16.0,'red pimples':17.0,'infected hair follicles':18.0,'Discharge or stickiness':19.0,'Small red, tender bumps':20.0}
    int_features = [res1[x] for x in request.form.values()]
    print(int_features)
    final_features = [np.array(int_features)]
    print(final_features)
    prediction = model.predict(final_features)
    output = abs(int(prediction[0]))
    res={0:'<p>Predicted Disease is Eczema<br><br>Treatment:Lotions and creams to keep skin moist, these should be applied when the skin is damp</p>',1:'<p>Predicted Disease is Tinea versicolo<br><br>Treatment:Fluconazole ,150mg once a week for 4 weeks</p>',2:'<p>Predicted Disease is Tinea faciei<br><br>Treatment:Terbinafine, Itraconazole 4-6 weeks plus topical</p>',3:'<p>Predicted Disease is Melanoma<br><br>Treatment:radiation treatments</p>',4:'<p>Predicted Disease is vitiligo<br><br>Treatment:surgical treatment</p>',5:'<p>Predicted Disease is acne<br><br>Treatment:treated with creams</p>'}
    output=res[output]
    output=Markup(output)
    return render_template('HTMLskin.html', prediction_text=output)

@app.route('/results',methods=['POST'])
def results():

    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)
