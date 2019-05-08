### The purpose of this module is to extract the relevant features from the given
### CSV file and preprocess the data so we get the feature matrix. We need:
### 1) Training Data
### 2) Training Labels
### 3) Test Data
### 4) Test Labels (for evaluation)
### After that, the data can be used to train and test any supervised classifier

import re
import csv
import random
import numpy

# word_bag_dictionary shows at which position a certain feature is in the wordBag.
# It holds a unique value for each word so that it's identifiable in the wordBag.
word_bag_dictionary = {}
word_bag = []

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


def __read_csv(file, debug=False):
    """Reads the CSV file with the data, splits it with the delimiter semicolon
    and returns the values as a list
    """
    csv_file = open(file, 'r')
    reader = csv.reader(csv_file, delimiter=';')
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
    if debug:
        return x
    return content


def __split_text(text, count_duplicates=False):
    """Splits all the tokens in a text. Returns them in a unique list. 
    If countDuplicates is true, the list can contain tokens multiple times
    """
    tokens = text.split(' ')
    if not count_duplicates:
        tokens = set(tokens)

    r = []
    for token in tokens:
        token = token.upper()
        token = __remove_special_characters_and_numbers(token)
        r.append(token)
    return r


def __extract_all_tokens(csv_content, count_duplicates=False):
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
    all_tokens = []
    for line in csv_content:
        # line[4] = Buchungstext, line[5] = Verwendungszweck, line[6] = Begünstigter
        tokens = __split_text(line[4] + " " + line[5] + " " + line[6], count_duplicates)
        all_tokens.append(tokens)
    return all_tokens


def __create_word_bag_and_dictionary(all_tokens):
    for token_line in all_tokens:
        for token in token_line:
            if token == '':
                continue
            if token not in word_bag_dictionary:
                word_bag.append(token)
                word_bag_dictionary[token] = len(word_bag) - 1


def __create_feature_matrix(all_tokens):
    """Creates the feature matrix based on all the tokens that were extracted.
    This function just creates the feature matrix based on the tokens but not on the 
    additional features like budget value or something else. However, it creates the
    array and adds extra space for these features. 
    """
    feature_matrix = []
    for token_line in all_tokens:
        # +2 because we still need to add the budget value and the labels
        feature_matrix.append([0] * (len(word_bag) + 2))
        for token in token_line:
            if token == '':
                continue
            pos = word_bag_dictionary[token]
            feature_matrix[len(feature_matrix) - 1][pos] += 1
    return feature_matrix


def __add_budget_value_and_labels(csv_content, feature_matrix):
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
        feature_matrix[i][len(feature_matrix[i]) - 2] = line[9].replace(",", ".")
        feature_matrix[i][len(feature_matrix[i]) - 1] = __get_label_nr(line[11].upper())
        i += 1
    return feature_matrix


# Function taken from https://machinelearningmastery.com/naive-bayes-classifier-scratch-python/
def __split_data_set(data_set, split_ratio):
    """Splits the dataset randomly into a training and a test set.
    The ratio is a value between 0 and 1 and describes the percentage on 
    how much data should be training data. For example, setting it to 0.8
    means we have 80 percent of training and 20 percent of testing data.
    """
    train_size = int(len(data_set) * split_ratio)
    train_set = []
    copy = list(data_set)
    while len(train_set) < train_size:
        index = random.randrange(len(copy))
        train_set.append(copy.pop(index))
    return [train_set, copy]


# return a tuple which contains the training data, the corresponding labels and the test data
def get_training_labels_and_test_data(percentage_of_training_data):
    """Returns a tuple of training data, training labels, test data and test labels
    Test labels are just returned for evaluating how good the classification worked
    """
    csv_content = __read_csv('Exercise 1 - Transaction Classification - Data Set.csv')
    all_tokens = __extract_all_tokens(csv_content, True)
    __create_word_bag_and_dictionary(all_tokens)
    feature_matrix = __create_feature_matrix(all_tokens)
    feature_matrix = __add_budget_value_and_labels(csv_content, feature_matrix)

    training_data_with_labels, test_data = __split_data_set(feature_matrix, percentage_of_training_data)
    training_data_with_labels = numpy.array(training_data_with_labels).astype(numpy.float)
    test_data = numpy.array(test_data).astype(numpy.float)

    test_labels = test_data[:, -1]
    # remove labels
    test_data = numpy.delete(test_data, len(test_data[0]) - 1, axis=1)
    training_labels = training_data_with_labels[:, -1]
    # remove labels
    training_data = numpy.delete(training_data_with_labels, len(training_data_with_labels[0]) - 1, axis=1)
    return training_data, training_labels, test_data, test_labels


def __get_label_nr(labelName):
    return LABELMAP.get(labelName, -1)
