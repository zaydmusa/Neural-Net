import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os
dir_path = os.path.dirname(os.path.realpath(__file__))

# layer sizes / parameters to adjust to find optimal accuracy & efficiency (parameters here yield ~85% accuracy)
inl = 784
hl1 = 512
hl2 = 128
outl = 10
learningRate = 0.1
iterations = 1000

classes	= 10

# Import data sets
testData = pd.read_csv(dir_path + '/MNIST-Datasets/fashion-mnist_test.csv')
trainData = pd.read_csv(dir_path + '/MNIST-Datasets/fashion-mnist_train.csv')

# Convert, transpose, & normalize
testData = np.array(testData).T
trainData = np.array(trainData).T
testLabel = testData[0]
testValues = testData[1:] / 255
trainLabel = trainData[0]
trainValues = trainData[1:] / 255

# Store key for dictionary
classKeys = {0:"t-shirt/top", 1:"trouser", 2:"pullover", 3:"dress", 4:"coat", 5:"sandal", 6:"shirt", 7:"sneaker", 8:"bag", 9:"ankle boot"}

# Dimensions
m, n = trainValues.shape
# print(m, n)

# Initializing weights and biases through random generation within given range (based on number of neurons, data size, etc.)
def initParameters():
	W1 = np.random.rand(hl1, m) - 0.5
	b1 = np.random.rand(hl1, 1) - 0.5
	W2 = np.random.rand(hl2, hl1) - 0.5
	b2 = np.random.rand(hl2, 1) - 0.5
	W3 = np.random.rand(classes, hl2) - 0.5
	b3 = np.random.rand(classes, 1) - 0.5
	return W1, b1, W2, b2, W3, b3

# ReLU function -- return Z if Z > 0, otherwise 0 -- that will be applied to get both hidden layers
def ReLU(Z):
	return np.maximum(Z, 0)

# Derivative of ReLU function -- constant bc same shape as y = x when x > 0 (1 if Z > 0, otherwise 0)
def ReLUDeriv(Z):
	return Z > 0

# Softmax function to apply to 2nd hidden layer to get output layer (get probability for each class)
def softmax(Z):
	expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))  # numeric stability
	return expZ / np.sum(expZ, axis=0, keepdims=True)

# Forward propagation to apply weights and biases to data to get pre and post-activation layers
def forwardProp(W1, b1, W2, b2, W3, b3, X):
	Z1 = W1.dot(X) + b1
	A1 = ReLU(Z1)
	Z2 = W2.dot(A1) + b2
	A2 = ReLU(Z2)
	Z3 = W3.dot(A2) + b3
	A3 = softmax(Z3)

	return Z1, A1, Z2, A2, Z3, A3

# Convert array to one-hot vector
def oneHot(Y):
	oneHotY = np.zeros((Y.size, int(Y.max())+1))
	oneHotY[np.arange(Y.size), Y] = 1
	return oneHotY.T

# Reverse engineer to find derivative of weights and biases in each connection
def backwardProp(Z1, A1, Z2, A2, Z3, A3, W1, W2, W3, X, Y):
	Yoh = oneHot(Y)
	dZ3 = A3 - Yoh
	dW3 = (1/n) * dZ3.dot(A2.T)
	db3 = (1/n) * np.sum(dZ3, axis=1, keepdims=True)

	dA2 = W3.T.dot(dZ3)
	dZ2 = dA2 * ReLUDeriv(Z2)
	dW2 = (1/n) * dZ2.dot(A1.T)
	db2 = (1/n) * np.sum(dZ2, axis=1, keepdims=True)

	dA1 = W2.T.dot(dZ2)
	dZ1 = dA1 * ReLUDeriv(Z1)
	dW1 = (1/n) * dZ1.dot(X.T)
	db1 = (1/n) * np.sum(dZ1, axis=1, keepdims=True)

	return dW1, db1, dW2, db2, dW3, db3

# Update the parameters using the derivatives from backwards propagation, considering the given learning rate
def updateParameters(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, alpha):
	W1 -= alpha * dW1;  b1 -= alpha * db1
	W2 -= alpha * dW2;  b2 -= alpha * db2
	W3 -= alpha * dW3;  b3 -= alpha * db3
	return W1, b1, W2, b2, W3, b3

# Give an array with the prediction for each image (class with highest given probability from softmax)
def getPredictions(A):
	return np.argmax(A, 0)

def getClothingItemPrediction(guess):
	return classKeys[guess]

# Compare the predictions array with the labels to find how many were calculated correctly from the total data set
def getAccuracy(predictions, Y):
	print(predictions, Y)
	return np.sum(predictions == Y) / Y.size

# Conduct gradient descent / train the model using forward and backward propagation through the data for however many given iterations
def gradientDescent(X, Y, alpha, iterations):
	W1, b1, W2, b2, W3, b3, = initParameters()
	for i in range(iterations):
		# Adjust learning rate in progression -- redundant feature
		# if i > 100 and i < 250:
		# 	alpha = alpha / 2
		# elif i >= 250:
		# 	alpha = alpha / 5

		Z1, A1, Z2, A2, Z3, A3, = forwardProp(W1, b1, W2, b2, W3, b3, X)
		dW1, db1, dW2, db2, dW3, db3 = backwardProp(Z1, A1, Z2, A2, Z3, A3, W1, W2, W3, X, Y)
		W1, b1, W2, b2, W3, b3 = updateParameters(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, alpha)
		if i % 10 == 0:
			print("Iteration: ", i)
			predictions = getPredictions(A3)
			print(getAccuracy(predictions, Y))
	return W1, b1, W2, b2, W3, b3

# Test the model using the parameters found from the training against a different data set (to prevent overfitting the training data)
def testModel(W1, b1, W2, b2, W3, b3, values, labels):
	_, _, _, _, _, A3 = forwardProp(W1, b1, W2, b2, W3, b3, values)
	predictions = getPredictions(A3)
	print()
	print("Testing against test data...")
	accuracy = getAccuracy(predictions, labels)
	print("Test data accuracy:", accuracy)
	return predictions

# Show the prediction and actual value for a given image from a given data set, and show the image to analyze why the model may have struggled or succeeded
def showPrediction(index, values, label, predictions):
	prediction = predictions[index]
	actual = label[index]
	img = values[:, index].reshape(28, 28)
	print(f"Index {index}, Predicted: {prediction}, Actual: {actual}")
	plt.imshow(img, cmap='gray')
	plt.axis('off')
	plt.show()

# training the model
# W1, b1, W2, b2, W3, b3 = gradientDescent(trainValues, trainLabel, learningRate, iterations)
# testing the model against test data to ensure it's not overfitting our training data
# predictions = testModel(W1, b1, W2, b2, W3, b3, testValues, testLabel)

# show some of the test images to see where the model might struggle
# showPrediction(1, testValues, testLabel, predictions)
# showPrediction(2, testValues, testLabel, predictions)
# showPrediction(3, testValues, testLabel, predictions)
# showPrediction(-3, testValues, testLabel, predictions)
# showPrediction(-2, testValues, testLabel, predictions)
# showPrediction(-1, testValues, testLabel, predictions)


#Storing the most accurate training model's weights and biases to reuse w/o having to rerun the gradient descent
# pd.DataFrame(W1).to_csv(dir_path + '/Model-Data/W1.csv', header=False, index=False)
# pd.DataFrame(b1).to_csv(dir_path + '/Model-Data/b1.csv', header=False, index=False)
# pd.DataFrame(W2).to_csv(dir_path + '/Model-Data/W2.csv', header=False, index=False)
# pd.DataFrame(b2).to_csv(dir_path + '/Model-Data/b2.csv', header=False, index=False)
# pd.DataFrame(W3).to_csv(dir_path + '/Model-Data/W3.csv', header=False, index=False)
# pd.DataFrame(b3).to_csv(dir_path + '/Model-Data/b3.csv', header=False, index=False)