import featureTransformation as ft
from sklearn.naive_bayes import GaussianNB

if __name__ == '__main__':
    nrOfFivePercentRuns = 5
    percentage = 0.05
    formatter = "{0.2f}"
    while percentage <= 0.80:
        X_Training, Y_Training, X_Test, Y_Test = ft.getTrainingLabelsAndTestData(percentage)
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

        print("Training data in percent: " + str(formatter.format(percentage)), end="\t")
        print("Correct: " + str(correctClassifications), end="\t")
        print("Incorrect: " + str(missClassifications), end="\t")
        print("Accuracy: " + str(formatter.format(correctClassifications / len(results))))
        nrOfFivePercentRuns -= 1
        if nrOfFivePercentRuns < 1:
            percentage += 0.05