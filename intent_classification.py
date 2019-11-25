import numpy as np
import json
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
import pickle
f=open('data.json','r')
j_file=json.loads(f.read())
dic={'intent':[],'text':[]}
for data in j_file['rasa_nlu_data']['common_examples']:
    dic['intent'].append(data['intent'])
    dic['text'].append(data['text'])

df=pd.DataFrame(dic)

vector_file=open('tfidf.pkl','wb')
encoder_file=open('encoder.pkl','wb')
c_vectorizer=CountVectorizer()
c_features=c_vectorizer.fit_transform(list(df['text']))
t_vectorizer = TfidfVectorizer()
t_features=t_vectorizer.fit_transform(list(df['text']))
print(t_features.shape,c_features.shape)
Training_Labels=df['intent']
print(len(np.unique(Training_Labels)))
encoder = LabelEncoder()
# Training labels will have 0,1,2 as value
encoded_Y=encoder.fit_transform(Training_Labels)
# convert integers to dummy variables (i.e. one hot encoded)
# print(set(encoded_Y))
# a=np.zeros(118)
# a[3]=1
# print(a)
# print(encoder.inverse_transform([0]))
dummy_y = np_utils.to_categorical(encoded_Y)
print(dummy_y.shape)
pickle.dump(t_vectorizer,vector_file)
pickle.dump(encoder,encoder_file)
model = Sequential()
model.add(Dense(300, input_shape=(c_features.shape[-1],)))
model.add(Dense(400))
model.add(Dense(dummy_y.shape[-1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

model.fit(t_features,dummy_y,epochs=100,batch_size=20)
model.save('intent_classifier.h5')
out=list(model.predict(t_vectorizer.transform(['hello ?']))[0])
# print(out)
# print(max(out))
i=out.index(max(out))
print(encoder.inverse_transform([i]))
