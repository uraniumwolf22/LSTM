#this is the testing branch
import random
from functions import *
def main():  #can take this out of a function if needed for UI
    #Begin program    
    print("Beginning")
    iterations = 25  #Change to change how many times the program loops through
    learningRate = 0.1
    returnData, numCategories, expectedOutput, outputSize, data = LoadText()
    print("Done Reading")
    RNN = RecurrentNeuralNetwork(numCategories, numCategories, outputSize, expectedOutput, learningRate)
    for i in range(1, iterations):

        RNN.forwardProp()
        error = RNN.backProp()
        mess = RNN.backProp()
        print("Error on iteration ",i, ": ", error)
        if error > -10 and error < 10 or i % 10 == 0:
            seed = np.zeros_like(RNN.x)
            maxI = np.argmax(np.random.random(RNN.x.shape))
            seed[maxI] = 1
            RNN.x = seed
            output = RNN.sample()
            ExportText(output, data)
    print("Complete")

main()