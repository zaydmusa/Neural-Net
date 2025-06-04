import neuralNet as neuralNet
import numpy as np
import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt
import os
dir_path = os.path.dirname(os.path.realpath(__file__))

# Convert to 28x28, then B&W, then to CSV
def parseImage(filepath, csvName):
	img = Image.open(filepath)
	img_resized = img.resize((28, 28))
	img_gray = img_resized.convert('L')

	# Display the original image for reference
	print("Now showing your original image.")
	plt.imshow(img)
	plt.axis('off')
	plt.show()

	# Display the adjusted image for reference
	print("Now showing your processed image.")
	plt.imshow(img_gray, cmap='gray')
	plt.axis('off')
	plt.show()

	pixels = list(img_gray.getdata())

	df = pd.DataFrame(pixels)
	df.to_csv(dir_path + '/Image-Uploads/imageData.csv', index=False, header=False)
	return (dir_path + '/Image-Uploads/imageData.csv')

# Now that we have a trained model to use, we can import it here
W1 = np.array(pd.read_csv(dir_path + '/Model-Data/W1.csv', header=None, index_col=False))
b1 = np.array(pd.read_csv(dir_path + '/Model-Data/b1.csv', header=None, index_col=False))
W2 = np.array(pd.read_csv(dir_path + '/Model-Data/W2.csv', header=None, index_col=False))
b2 = np.array(pd.read_csv(dir_path + '/Model-Data/b2.csv', header=None, index_col=False))
W3 = np.array(pd.read_csv(dir_path + '/Model-Data/W3.csv', header=None, index_col=False))
b3 = np.array(pd.read_csv(dir_path + '/Model-Data/b3.csv', header=None, index_col=False))

# Get and adjust the image
imagePath = input("Hey! To test the network against your image, please: \n1. Ensure the background is dark \n2. Crop the image to a square ratio \n3. Ensure the item's fully in frame \n4. Upload it to the Image-Uploads folder \nNow you're all set, just copy and paste the FULL file path here and I'll take it from there: ")
imagePixelPath = parseImage(imagePath, "imagePixels")
imagePixels = np.array(pd.read_csv(imagePixelPath, header=None, index_col=False))

# Get A3 (probabilities of each class) from image
_, _, _, _, _, A3 = neuralNet.forwardProp(W1, b1, W2, b2, W3, b3, imagePixels/255)

# Now ask if the guess was correct, and if not, guess with the next most sure answer
guessAccuracy = False
guesses = 0
while not guessAccuracy:
	guess = neuralNet.getPredictions(A3)[0]
	clothingItem = neuralNet.getClothingItemPrediction(guess)
	guessInput = input("Was your image a " + clothingItem + "? (Enter Y/N): ")
	guesses += 1

	if guessInput == 'Y' or guessInput == 'y':
		guessAccuracy = True
	else:
		guessAccuracy = False

	if guessAccuracy == False:
		"Okay, let me try again"
		A3[guess] = 0
		continue

print("Nice! The model figured out your item in " + str(guesses) + " guess(es).")