# Train model and make predictions
import numpy
import pandas
from keras.utils import np_utils
from sklearn import datasets
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from pickle import dump

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# load dataset
dataframe = pandas.read_csv("training_data.csv", header=0)
dataset = dataframe.values
X = dataset[:, 0:6].astype(float)
Y = dataset[:, 6]

# process the dataset to fit in the fitting
# X = preprocessing.scale(X)
scaler = MinMaxScaler()
scaler.fit(X)
X = scaler.transform(X)
dump(scaler, open('scaler.pkl', 'wb'))
print("saved the scaler")

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)

# convert integers to dummy variables (i.e. one hot encoded)
y = np_utils.to_categorical(encoded_Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.3, random_state=seed)
