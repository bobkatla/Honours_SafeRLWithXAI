import pandas
import numpy

from model_loading import hazard_prediction, model

# load dataset
dataframe = pandas.read_csv("testing.csv", header=0)
dataset = dataframe.values
X_test = dataset[:, 0:6].astype(float)
Y_test = dataset[:, 6]
n = len(Y_test)

n0, n1, n2, n3 = 0, 0, 0, 0
a0, a1, a2, a3 = 0, 0, 0, 0
for i in range(n):
    temp, humid, wall, x, y, action = X_test[i]
    predict = hazard_prediction(temp, humid, wall, x, y, action)
    print(f'testing {i}')
    if Y_test[i] == 0:
        n0 += 1
        if predict == Y_test[i]:
            a0 += 1
    elif Y_test[i] == 1:
        n1 += 1
        if predict == Y_test[i]:
            a1 += 1
    elif Y_test[i] == 2:
        n2 += 1
        if predict == Y_test[i]:
            a2 += 1
    elif Y_test[i] == 3:
        n3 += 1
        if predict == Y_test[i]:
            a3 += 1

print(0, a0, n0, (a0*100)/n0)
print(1, a1, n1, (a1*100)/n1)
print(2, a2, n2, (a2*100)/n2)
print(3, a3, n3, (a3*100)/n3)
