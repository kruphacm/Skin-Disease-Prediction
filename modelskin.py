import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
df=pd.read_csv('https://raw.githubusercontent.com/kruphacm/mini-project/main/Data%20of%20skin.csv')
df['Symptom 1'] = df['Symptom 1'].map({'Itching':0.0,'Skin becomes dry, thickened and scaly':1.0,'Rash-commonly affect face, neck, back of knees and arms':2.0,'Fine, branny scaling':3.0,'Upper trunk, upper arms, neck and abdomen':4.0,'change of skin color, white patches':5.0,'skin peeling':6.0,'Vesicular, pustular':7.0,'Dry type infection':8.0,'moles have asymmetrical shapes':9.0,'uneven colors of moles':10.0,'change in size of moles':11.0,'loss of hair color':12.0,"patches of skin":13.0,"pigmentation in the skin":14.0,'oval macules':15.0,"red pimples":16.0,"infected hair follicles":17.0,"painful lumps":18.0,'Small red, tender bumps':19.0})

df['Symptom 2'] = df['Symptom 2'].map({'Itching':0.0,'Skin becomes dry, thickened and scaly':1.0,'Rash-commonly affect face, neck, back of knees and arms':2.0,'Fine, branny scaling':3.0,'Upper trunk, upper arms, neck and abdomen':4.0,'change of skin color, white patches':5.0,'skin peeling':6.0,'Vesicular, pustular':7.0,'Dry type infection':8.0,'moles have asymmetrical shapes':9.0,'uneven colors of moles':10.0,'change in size of moles':11.0,'loss of hair color':12.0,"patches of skin":13.0,"pigmentation in the skin":14.0,'oval macules':15.0,"red pimples":16.0,"infected hair follicles":17.0,"painful lumps":18.0,'Small red, tender bumps':19.0})

df['Symptom 3'] = df['Symptom 3'].map({'Itching':0.0,'Skin becomes dry, thickened and scaly':1.0,'Rash-commonly affect face, neck, back of knees and arms':2.0,'Fine, branny scaling':3.0,'Upper trunk, upper arms, neck and abdomen':4.0,'change of skin color, white patches':5.0,'skin peeling':6.0,'Vesicular, pustular':7.0,'Dry type infection':8.0,'moles have asymmetrical shapes':9.0,'uneven colors of moles':10.0,'change in size of moles':11.0,'loss of hair color':12.0,"patches of skin":13.0,"pigmentation in the skin":14.0,'oval macules':15.0,"red pimples":16.0,"infected hair follicles":17.0,"painful lumps":18.0,'Small red, tender bumps':19.0})

X = df.drop(['Disease','Treatment'], axis=1)
df['Disease'] = df['Disease'].map({'Eczema':0.0,'Tinea versicolo':1.0,'Tinea faciei':2.0,'Melanoma':3.0,'vitiligo':4.0,'acne':5.0})
Y = df['Disease']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33)
model1=LogisticRegression()
model1.fit(X_train,y_train)
pickle.dump(model1, open('modelskin.pkl','wb'))

model = pickle.load(open('modelskin.pkl','rb'))