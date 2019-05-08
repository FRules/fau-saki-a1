### The purpose of this module is to extract the relevant features from the given
### CSV file and preprocess the data so we get the feature matrix. We need:
### 1) Training Data
### 2) Training Labels
### 3) Test Data
### 4) Test Labels (for evaluation)
### After that, the data can be used to train and test any supervised classifier

import re
from io import StringIO
import csv
import random
import numpy

# wordbagDictionary shows at which position a certain feature is in the wordBag. 
# It holds a unique value for each word so that it's identifiable in the wordBag.
wordBagDictionary = {}
wordBag = []

LABELMAP = {
    "INCOME": 1,
    "LIVING": 2,
    "PRIVATE": 3,
    "STANDARDOFLIVING": 4,
    "LEISURE": 5,
    "FINANCE": 6,
    "TRAFFIC": 7
}

def __remove_special_characters_and_numbers(str):
    """Removes all special characters and numbers from a string
    If we don't do this, we end up with a ton of feature vectors that are not
    representive, because e.g. Kundenreferenznummer12345 would end up in 
    just a few samples but the better feature, Kundenreferenznummer, would end up
    in far more samples.
    """
    str = re.sub('[\W_]+', '', str)
    str = re.sub('[\d_]+', '', str)
    return str;

def __readCsv(file, debug = False):
    """Reads the CSV file with the data, splits it with the delimiter semicolon
    and returns the values as a list
    """
    csvfile = open(file, 'r')
    reader = csv.reader(csvfile, delimiter=';')
    content = list(reader)
    # Remove header
    content.pop(0)
    # for test purposes, I implemented a switch to return just
    # the first 5 lines
    x = []
    x.append(content[0])
    x.append(content[1])
    x.append(content[2])
    x.append(content[3])
    x.append(content[4])
    if debug == True:
        return x
    return content

def __splitText(text, countDuplicates = False):
    """Splits all the tokens in a text. Returns them in a unique list. 
    If countDuplicates is true, the list can contain tokens multiple times
    """
    tokens = text.split(' ')
    if countDuplicates == False:
        tokens = set(tokens)

    r = []
    for token in tokens:
        token = token.upper()
        token = __remove_special_characters_and_numbers(token)
        r.append(token)
    return r

def __extractAllTokens(csv_content, countDuplicates = False):
    """Extracts all tokens from the string cells in the csv file
    Tokens are single words. Every token has to be in the feature vector.
    In text classification, if we had 200 words, we'd have a 200-dimensional
    feature vector.
    We're only interested in the following features in the dataset:
    Buchungstext, Verwendungszweck, Beguenstigter/Zahlungspflichtiger, Betrag, Label
    Since the first three columns are text, we cant just use the data as it is,
    we need to transform it - we need a bag of words as features in addition 
    to the Betrag-column. For example, if we have the following two columns...
    [
      [Lohn / Gehalt; Gehalt Adorsys GmbH & Co. KG End-To-End-Ref.: Notprovided Kundenreferenz: Nsct1603300013660000000000000000001 Gutschrift; Adorsys GmbH & Co. KG; 2000.00; income ]
      [Miete;	Byladem1Sbt De12773501123456789889 Miete Beuthener Str. 25 End-To-End-Ref.: Notprovided Dauerauftrag Dauerauftrag; Georg Tasche; -670.00; living ]
    ]
    ... the feature vector (matrix) should look like this
    [
      [Lohn: 1, Gehalt: 2, Adorsys: 2, GmbH: 2, Co: 2, KG: 2, EndToEndRef: 1, Notprovided: 1, Kundenreferenz: 1, Nsct: 1,
       Gutschrift: 1, Miete: 0, Bylademsbt: 0, DE: 0, Beuthener: 0, Str: 0, Dauerauftrag: 0, Georg: 0, Tasche: 0, Betrag: 2000, Label: 1]
      [Lohn: 0, Gehalt: 0, Adorsys: 0, GmbH: 0, Co: 0, KG: 0, EndToEndRef: 1, Notprovided: 1, Kundenreferenz: 0, Nsct: 0, 
       Gutschrift: 0, Miete: 2, Bylademsbt: 1, DE: 1, Beuthener: 1, Str: 1, Dauerauftrag: 2, Georg: 1, Tasche: 1, Betrag: -670, Label: 2]
    ]
    """
    allTokens = []
    for line in csv_content:
		#line[4] = Buchungstext, line[5] = Verwendungszweck, line[6] = Begünstigter
        tokens = __splitText(line[4] + " " + line[5] + " " + line[6], countDuplicates)
        allTokens.append(tokens)
    return allTokens

def __createWordBagAndDictionary(allTokens):
    for tokenLine in allTokens:
        for token in tokenLine:
            if token == '':
                continue
            if token not in wordBagDictionary:
                wordBag.append(token)
                wordBagDictionary[token] = len(wordBag) - 1

def __createFeatureMatrix(allTokens):
    """Creates the feature matrix based on all the tokens that were extracted.
    This function just creates the feature matrix based on the tokens but not on the 
    additional features like budget value or something else. However, it creates the
    array and adds extra space for these features. 
    """
    featureMatrix = []
    for tokenLine in allTokens:
        # +2 because we still need to add the budget value and the labels
        featureMatrix.append([0] * (len(wordBag) + 2))
        for token in tokenLine:
            if token == '':
                continue
            pos = wordBagDictionary[token]
            featureMatrix[len(featureMatrix)-1][pos] += 1
    return featureMatrix

def __addBudgetValueAndLabels(csv_content, featureMatrix):
    """Adds the budget value and the labels to the feature matrix.
    The feature matrix contains at this stage just the whole word dictionary
    as feature vectors. We still need the budget value for classifying tasks
    and the labels for the training and testing process, so this function
    adds them to the feature vectors. It maps the label to a number, specified
    by the LABELMAP properties
    """
    i = 0
    for line in csv_content:
        # budget value sometimes has commas in it so we cant convert it to float later on. 
        # therefor, we replace the comma with a point so we can cast it to float. 
        featureMatrix[i][len(featureMatrix[i])-2] = line[9].replace(",",".")
        featureMatrix[i][len(featureMatrix[i])-1] = __getLabelNr(line[11].upper())
        i += 1
    return featureMatrix

# Function taken from https://machinelearningmastery.com/naive-bayes-classifier-scratch-python/
def __splitDataset(dataset, splitRatio):
    """Splits the dataset randomly into a training and a test set.
    The ratio is a value between 0 and 1 and describes the percentage on 
    how much data should be training data. For example, setting it to 0.8
    means we have 80 percent of training and 20 percent of testing data.
    """
    trainSize = int(len(dataset) * splitRatio)
    trainSet = []
    copy = list(dataset)
    while len(trainSet) < trainSize:
        index = random.randrange(len(copy))
        trainSet.append(copy.pop(index))
    return [trainSet, copy]

#return a tuple which contains the training data, the corresponding labels and the test data
def getTrainingLabelsAndTestData(percentageOfTrainingData):
    """Returns a tuple of training data, training labels, test data and test labels
    Test labels are just returned for evaluating how good the classification worked
    """
    csv_content = __readCsv('Exercise 1 - Transaction Classification - Data Set.csv')
    allTokens = __extractAllTokens(csv_content, True)
    __createWordBagAndDictionary(allTokens)
    featureMatrix = __createFeatureMatrix(allTokens)
    featureMatrix = __addBudgetValueAndLabels(csv_content, featureMatrix)

    trainingDataWithLabels, testData = __splitDataset(featureMatrix, percentageOfTrainingData)
    trainingDataWithLabels = numpy.array(trainingDataWithLabels).astype(numpy.float)
    testData = numpy.array(testData).astype(numpy.float)

    testLabels = testData[:,-1]
    # remove labels
    testData = numpy.delete(testData, len(testData[0]) - 1, axis=1)
    trainingLabels = trainingDataWithLabels[:,-1]
    # remove labels
    trainingData = numpy.delete(trainingDataWithLabels, len(trainingDataWithLabels[0]) - 1, axis=1)
    return trainingData, trainingLabels, testData, testLabels

def __getLabelNr(labelName):
    return LABELMAP.get(labelName, -1)