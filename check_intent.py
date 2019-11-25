from keras.models import load_model
import pickle
vectorizer = pickle.load(open("tfidf.pkl", "rb"))
encoder = pickle.load(open("encoder.pkl", "rb"))
clf=load_model('intent_classifier.h5')
clf._make_predict_function()
print('enter "q" to exit...')
while True:
    text=input('Enter some input: ')
    output=list(clf.predict(vectorizer.transform([text]))[0])
    i=output.index(max(output))
    print('INTENT:',encoder.inverse_transform([i]))
