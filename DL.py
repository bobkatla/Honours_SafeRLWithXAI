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
dataframe = pandas.read_csv("./training_data.csv", header=0)
dataset = dataframe.values
X = dataset[:,0:6].astype(float)
Y = dataset[:,6]

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
	model.add(Dense(12, input_dim=6, activation='relu'))
	# model.add(Dense(24, activation='relu'))
	model.add(Dense(4, activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

estimator = KerasClassifier(build_fn=baseline_model, epochs=500, batch_size=5, verbose=0)

kfold = KFold(n_splits=10, shuffle=True)
results = cross_val_score(estimator, X, dummy_y, cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

# Need to fit first
estimator.fit(X, Y)
print("hey")

# serialize model to JSON
model_json = estimator.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
estimator.save_weights("model.h5")
print("Saved model to disk")

# Create a new dataset
# Save the model to run later on the other file (no import directly)
# Try to find the way for knowing the final weight in the last layer

# func: input robot pos, output feature
# create env, maybe actually may a spreading value from a hazard (like fire will affect the 3 near squares )
# manual work for starting to gather the data for DL training

