import numpy
import pandas
from keras.models import model_from_json
from data_preparation import X_test, Y_test, X_train, Y_train
from pickle import load


def load_model():
    # loading model
    model = model_from_json(open('model_architecture.json').read())
    model.load_weights('model_weights.h5')
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model


# load
model = load_model()


def hazard_prediction(temp, humid, wall, x, y, action):
    # load the scaler
    scaler = load(open('scaler.pkl', 'rb'))
    prep = numpy.array([[temp, humid, wall, x, y, action]])
    prep = scaler.transform(prep)
    prediction = model.predict(prep, verbose=0)
    max_val = max(prediction[0])
    final_result = numpy.where(prediction[0] == max_val)
    return final_result[0][0]


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