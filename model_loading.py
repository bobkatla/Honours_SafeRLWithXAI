import numpy
import pandas
from keras.models import model_from_json
from sklearn import preprocessing
from data_preparation import X_test, Y_test, X_train, Y_train
from sklearn.model_selection import train_test_split


def load_model():
    # loading model
    model = model_from_json(open('model_architecture.json').read())
    model.load_weights('model_weights.h5')
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model


# load
model = load_model()

def hazard_prediction(temp, humid, wall, x, y, action):
    prep = numpy.array([[temp, humid, wall, x, y, action]])
    # prep = preprocessing.scale(prep)
    prediction = model.predict(prep, verbose=0)
    max_val = max(prediction)
    final_result = numpy.where(prediction == max_val)
    return final_result[0][0]

# prep = numpy.array([[0,100.0,0.0,7.0,3.0,2.0]])
# prep = preprocessing.scale(prep)
# print(prep)

# print(hazard_prediction(80,100.0,0.0,7.0,3.0,2.0))

# predictions
# Work with this custom data for test

df = pandas.read_csv("./training_data.csv", header=0)
dataset = df.values
testu = dataset[:5, 0:6].astype(float)
print(testu)

testu = preprocessing.scale(testu)
print(testu)

# the same
# check = model.predict(testu, verbose=0)
# for x in check:
#     max_val = max(x)
#     final_result = numpy.where(x == max_val)
#     print(final_result[0][0])
'''
predictions = model.predict(X_train, verbose=0)
# This prediction is the final weights!
print(predictions)
result = []
for pred in predictions:
    max_val = max(pred)
    final_result = numpy.where(pred == max_val)
    result.append(final_result[0][0])

check = 0
for i in range(len(Y_train)):
    max_val = max(Y_train[i])
    final_result = numpy.where(Y_train[i] == max_val)
    final_val = final_result[0][0]
    if final_val == result[i]:
        check += 1

print(result)
print('The accuracy is: ')
print(check / len(Y_train))
print(len(Y_train) - check)
print(len(X_train))
'''