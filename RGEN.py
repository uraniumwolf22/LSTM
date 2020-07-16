#TODO:  Build UI


import random
from functions import *

#Begin program    
print("Beginning")                                                          
iterations = 25                                                             #Change to change how many times the program loops through needs to be adjustable in UI
learningRate = 0.1                                                          #Learning rate Needs to be adjustable in UI
returnData, numCategories, expectedOutput, outputSize, data = LoadText()
print("Done Reading")                                                       #would be nice if the console window could also be built into the UI

RNN = RecurrentNeuralNetwork(numCategories, numCategories, outputSize, expectedOutput, learningRate) #Initialize Recurrent Neural Network
for i in range(1, iterations):

    RNN.forwardProp()                                                       #Forward propigate
    error = RNN.backProp()                                                  #define error and mess
    mess = RNN.backProp()
    print("Error on iteration ",i, ": ", error)                             #This should also be in the UI console window
    if error > -10 and error < 10 or i % 10 == 0:
        seed = np.zeros_like(RNN.x)
        maxI = np.argmax(np.random.random(RNN.x.shape))
        seed[maxI] = 1
        RNN.x = seed
        output = RNN.sample()
        ExportText(output, data)
print("Complete")
