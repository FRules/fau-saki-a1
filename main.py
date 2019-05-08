import featureTransformation as ft
from sklearn.naive_bayes import GaussianNB

if __name__ == '__main__':
    X_Training, Y_Training, X_Test, Y_Test = ft.get_training_labels_and_test_data(0.67)
    classificator = GaussianNB()
    classificator.fit(X_Training, Y_Training)
    results = classificator.predict(X_Test)

    i = 0
    missClassifications = 0
    correctClassifications = 0
    for result in results:
        if result == Y_Test[i]:
            correctClassifications += 1
        else:
            missClassifications += 1
        i += 1

    print("Correct: " + str(correctClassifications))
    print("Incorrect: " + str(missClassifications))
    print("Accuracy: " + str(correctClassifications / len(results)))
