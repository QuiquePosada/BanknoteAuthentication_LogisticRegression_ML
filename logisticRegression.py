""" 
Enrique Posada Lozano 
A01700711
ITESM Campus QRO
Logistic Regression Algorithm
"""
import sys
import math
import numpy
import pandas
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score
import matplotlib.pyplot as plot
import time
import seaborn

from sklearn.linear_model import LogisticRegression


__errors__= []

def calculateHyp(params,sample):
    """
    Calculates the predicted value (hypothesis)
    yHat = 1.0 / (1.0 + e ^ (-(b0+b1*x1...+bn*xn)))     # Simplified Formula for Logistic Regression
    NOTE : sigmoid func does this, this func as is just does (-(b0+b1*x1...+bn*xn)) and then calls sigmoid
    yHat -> Predicted Value,    b0 -> Bias,     b1 -> Coefficient of 1st parameter ,    bn -> n coefficent 
    x1 -> first input (feature -> ex: skewness value),  xn -> n input/feature 

    params -> Coefficientes of each parameter
    sample -> instance contained in the features dataset
    """
    acc = 0
    acc = params * sample # pandas takes care of multiplying each parameter with the respective feature
    # acc = acc.to_numpy().sum() # Converts the dataframe calculated to an array, and then proceeds to add all the values
    #     # Basically, it first does all the multiplication and afterwards adds all the values up
    # acc = acc * (-1)

    # Optimized version
    acc = acc.sum(axis=1) # To sum by columns and not rows, axis is set to 1
    acc = acc * (-1)
    predictedValue = sigmoid(acc)
    # print("PREDICTIONS")
    # print(predictedValue)
    # time.sleep(5)
    return predictedValue

# def calculateHyp_Test(params,sample):
#     acc = 0
#     acc = params * sample
#     acc = acc.to_numpy().sum()
#     acc = 


def sigmoid(z):
    """
    Takes care of the activation function given z
    z -> (-(b0+b1*x1...+bn*xn))
    """
    # sigmoid = 1 / (1 + math.exp(z))
    sigmoid = 1 / (1 + numpy.exp(z)) # Exp with numpy works with pandas dataframes
    return sigmoid

def gradientDescent(params,features,learning_rate,expectedValues):
    """
    error = predictedValue - expectedValue
    """
    error = 0
    newParams = list(params)
    # for param in range(len(params)):
    #     sumErrors = 0 # Keeps an accumulate of the errors
    #     acc = 0 # coefficient value
    #     for instance in range(len(features)):
    #         yhat = calculateHyp(params,features.iloc[instance])
    #         error = yhat - expectedValues.iloc[instance]
    #         acc = acc + (error * features.iloc[instance,param]) # Calculate sumatory of gradient descent formula
    #         # acc = acc + (learning_rate * (expectedValues.iloc[instance] - yhat) * yhat * (1 - yhat) * )
    #     newParams[param] = params[param] - learning_rate * (1/len(features) * acc) # Here is the formula taught for gradient descent, acc is the value obtained from the sumatory
    
    acc = 0
    yHat = calculateHyp(params,features)
    error = yHat - expectedValues
    # print("MY ERROR")
    # print(error)
    # print("MY FEAT")
    # print(features)
    # error = error.to_numpy()
    # print("MY FEEEEEEEEEEAT")
    # print(numpy.dot(error,features))
    acc = numpy.dot(error,features)  # numpy takes care of all of this by calculating the dot product, thus getting the five parameters

    newParams = params - learning_rate * (1 / len(features) * acc)

    # print("NEW PARAMS")
    # print(newParams)

    # print("AHHHHH ACC")
    # print(acc)
    # time.sleep(10)
    return newParams

def show_errors(params, samples, y):
    """
	Calculates error (borrowed from Benji's implementation)
    """
    global __errors__
    error_acum = 0
    error = 0

    hyp = calculateHyp(params,samples)
    # print("MY HYP")
    # print(hyp)
    # print("MY Y")
    # print(y)
    # error = crossEntropy(hyp,y)
    error = numpy.vectorize(crossEntropy)(hyp,y)

    # print("PING0")
    # print(error)
    # print("COUNT and SUM!!")
    # print(str(len(error)) + " SUM -> " + str(error.sum()))
    error_acum = error.sum()
    # time.sleep(10)

    # for instance in range(len(samples)):
    #     hyp = calculateHyp(params,samples.iloc[instance])
    #     error = crossEntropy(hyp, y.iloc[instance])
    #     error_acum = error_acum + error # this error is different from the one used to update, this is general for each sentence it is not for each individual param
	#print("acum error %f " % (error_acum));
    mean_error_param = error_acum/len(samples)
	#print("mean error %f " % (mean_error_param));
    __errors__.append(mean_error_param)
    return mean_error_param

def crossEntropy(predictedValue, realValue):
    """
    Loss Function that is used to measure the performance of the classification model's predicted value
    Code is from on https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html 
    −(ylog⁡(p)+(1−y)log(1−p))  -> The original Math Formula for Cross-Entropy

    log is with base e (natural logarithm)

    Help from this resource -> conditionals with pandas https://guillim.github.io/pandas/2018/10/22/Pandas-if-else-on-columns.html
    """
    # print("CROSS ENTROOPY, WEWLCOME")
    # print(predictedValue)
    # print(realValue)
    # time.sleep(2)
    if realValue == 1: 
        if predictedValue == 0: # Just like in Benji's code, this prevents log(0)
            predictedValue = 0.001
        # return -(math.log(predictedValue))
        # print("RESULT")
        # print(-(numpy.log(predictedValue)))
        return -(numpy.log(predictedValue))
    else:
        if predictedValue == 1:
            predictedValue = 0.999
        # return -(math.log(1 - predictedValue))
        # print("RESULT")
        # print(-(numpy.log(1 - predictedValue)))
        return -(numpy.log(1 - predictedValue))

def scaleData(features):
    """
    Normalizes features in order for gradient descent to work correctly (improves the convergence speed of the logistic regression algorithm)
    features is an arg containing the sample of feature data to be normalized

    Normalization is made using Rescaling (min-max normalization) (https://en.wikipedia.org/wiki/Feature_scaling)

    returns the features of the dataset in a normalized manner

    normalizedVal = (x - min(x)) / (max(x) - min(x)))
    """
    # print("\nMAX OF DATASET")
    # print(features.max())
    maxValues = features.max()
    # print("\nMIN OF DATASET")
    # print(features.min())
    minValues = features.min()
    print("Initializing Normalization ...\n\n")
    for instance in range(len(features)):
        features.iloc[instance] = (features.iloc[instance] - minValues) / (maxValues - minValues)
    return features


# File name of the Dataset
csv_fileName = "data_banknote_authentication.csv"
# Get a dataframe from the Dataset provided and name each column of the dataset (since in this case data was not labeled within the file)
dataset = pandas.read_csv(csv_fileName,names=["Variance","Skewness","Curtosis","Entropy","Class"])

# Output the first 5 lines of the dataset
print(dataset.head())
print("\n\nDataset Description\n")
print(dataset.describe())

features = pandas.DataFrame(dataset, columns = ["Variance","Skewness","Curtosis","Entropy"])
label = dataset["Class"]

# Data has first to be normalized [0,1]
features = scaleData(features)
print("Normalized Features")
print(features)

authentic = dataset.loc[label == 1]

counterfeit = dataset.loc[label == 0]

# Learning Rate for GD
alpha = 3.5

# Starting Coefficients (For each parameter)
params = [0,0,0,0]

# Add a new column and param for the Bias
# NOTE THE MAIN PROBLEM AFTER EXECUTING THE CODE WAS THE MISSING BIAS, SINCE THE MINIMUM ERROR WOULD GO AROUND 0.35...
params.append(0) # Bias coefficient

# NOTE After giving the coefficients (parameters) random values, finally the learning adjusted and therefore, a learning rate 3.8 was too big

# Establish parameters starting value with random values
params = numpy.random.rand(len(params))
print("PARAMS starting values")
print(params)

print("NEW PARAMS")
print(params)
features["Bias"] = 1
print("NEW FEATURES with BIAS")
print(features.head())

# Splits the Dataset into a training dataset and a test dataset
    # In this case, the model is trained with 75% (3/4 of data) of the data given (NOTE: Since no random_state is given, the seed by default is random for each time the code is executed)
    # Registered amateur mistake -> used test_size instead of train_size, wondering why numbers did not match
trainingFeatures, testFeatures, trainingLabel, testLabel = train_test_split(features, label, train_size=0.25)
print("TRAIN features")
print(trainingFeatures)
print("TRAIN LABEL")
print(trainingLabel)

lr = LogisticRegression(C=1000.0, random_state=0)

lr.fit(trainingFeatures, trainingLabel)
y_pred = lr.predict(testFeatures)
print('Wrong/Misclassified samples: %d' % (testLabel != y_pred).sum())
print ("Accuracy of Model with Test Data (%) : ", accuracy_score(testLabel.values.tolist(), y_pred)) 

cm = confusion_matrix(testLabel.values.tolist(), y_pred)
print(cm) 


# Current epoch iteration
epoch = 0 

start_time = time.time()

predicted_Values = []

# While loop that stops until local minimum is reached or there is no further improvement in the bias
while True:
    prevParams = list(params) # previous epoch bias
    # if epoch % 128:
        # alpha = alpha * 0.5
    params = gradientDescent(params,trainingFeatures,alpha,trainingLabel)
    # params = coefficients_sgd(trainingFeatures, alpha, 100)
    error = show_errors(params, trainingFeatures, trainingLabel) # calculates the error between predicted and real data
    params = list(params) # In order to leave in same format as before -> not in a numpy array
    if(params == prevParams or epoch >= 10000 or error < 0.05): # the loop will only end if no further changes are made/seen in the params, the number of epochs given is reached or a given minimum error is reached
        # for instance in range(len(trainingFeatures)):
        #     yhat = calculateHyp(params,trainingFeatures.iloc[instance])
        #     predicted_Values.append(round(yhat))
        yHat = calculateHyp(params,trainingFeatures)
        yHat = yHat.to_numpy().round()
        # predicted_Values.append(round(yHat))
        predicted_Values = yHat
        # print("predicted values")
        # print(predicted_Values)
            # print("Expected -> %.3f , Predicted Value -> %.3f [%d]" % (trainingLabel.iloc[instance], yhat, round(yhat)))
        print ("FINAL params :")
        print (params)
        print("THE TRAINING HAS FINISHED IN " + str(epoch) + " EPOCHS!!")
        finishedTrainingTime = time.time() - start_time
        print("The training lasted for " + str(finishedTrainingTime/60) + " minutes")
        break
    epoch += 1
    # print(".",end="") # Prevents new line insertion at end of print
    print("EPOCHS -> " + str(epoch) + " and error -> " + str(error), end="\r") # Overwrites the current line

plot.plot(__errors__)
plot.title("Error")
plot.xlabel("# Epochs")
plot.ylabel("Error")
plot.show()

print ("Accuracy of Model with Training Data (%) : ", accuracy_score(trainingLabel.values.tolist(), predicted_Values)) 

cm = confusion_matrix(trainingLabel.values.tolist(), predicted_Values) 

correctPredictions = cm[0][0] + cm[1][1]
wrongPredictions = cm[0][1] + cm[1][0]
  
print ("Confusion Matrix : \n", cm) 
print("You have " + str(correctPredictions) + " of correct predictions and " + str(wrongPredictions) + " are wrong out of " + str(trainingFeatures.shape[0]))
plot.matshow(cm)
plot.title('Authentic vs Counterfeit Banknotes')
plot.colorbar()
plot.ylabel('Real Value')
plot.xlabel('Predicted Value')
plot.show()

print("\n\n\n\nEnd of Training the Model")

predicted_Values = []

# Here you just predict data of test values with params from training
print("Here is prediction for the test DATA :")
yhat = calculateHyp(params,testFeatures)
# predicted_Values.append(round(yhat))
predicted_Values = yhat.to_numpy().round()
for instance in range(len(testFeatures)):
    # yhat = calculateHyp(params,testFeatures.iloc[instance])
    # predicted_Values.append(round(yhat))
    print("Expected -> %.3f , Predicted Value -> %.3f [%d]" % (testLabel.iloc[instance], yhat.iloc[instance], round(yhat.iloc[instance])))

print ("Accuracy of Model with Test Data (%) : ", accuracy_score(testLabel.values.tolist(), predicted_Values)) 
cm = confusion_matrix(testLabel.values.tolist(), predicted_Values) 

correctPredictions = cm[0][0] + cm[1][1]
wrongPredictions = cm[0][1] + cm[1][0]
  
print ("Confusion Matrix : \n", cm) 
print("You have " + str(correctPredictions) + " of correct predictions and " + str(wrongPredictions) + " are wrong out of " + str(testFeatures.shape[0]))
