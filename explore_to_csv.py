import numpy
import pandas
from keras.models import model_from_json
from keras.layers import Dense
from keras.utils import np_utils
from sklearn import datasets
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from data_preparation import X_test, Y_test

def load_model():
    # loading model
    model = model_from_json(open('model_architecture.json').read())
    model.load_weights('model_weights.h5')
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

# load
model = load_model()

# predictions
predictions = model.predict(X_test, verbose=0)
# This prediction is the final weights!
# print(predictions)
result = []
for pred in predictions:
    max_val = max(pred)
    final_result = numpy.where(pred == max_val)
    result.append(final_result[0][0])

check = 0
for i in range(len(Y_test)):
    max_val = max(Y_test[i])
    final_result = numpy.where(Y_test[i] == max_val)
    final_val = final_result[0][0]
    if final_val == result[i]:
        check += 1

print('The accuracy is: ')
print(check / len(Y_test))

