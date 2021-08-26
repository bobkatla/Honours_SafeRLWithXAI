# multi-class classification with Keras
import pandas

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

# load dataset
dataframe = pandas.read_csv("./data.csv", header=0)
dataset = dataframe.values
X = dataset[:, 0:3].astype(float)
Y = dataset[:, 3]

encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)


# print(X)
# print(Y)

# define baseline model
def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(12, input_dim=3, activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# estimator = KerasClassifier(build_fn=baseline_model, epochs=500, batch_size=64, verbose=0)

# kfold = KFold(n_splits=10, shuffle=True)
# results = cross_val_score(estimator, X, dummy_y, cv=kfold)
# print("Baseline: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))

# Need to fit first
# estimator.fit(X, Y)
# print("hey")

# serialize model to JSON
# estimator = baseline_model()
# estimator.fit(X, Y)
# model.save('test_model')

# Try to find the way for knowing the final weight in the last layer
