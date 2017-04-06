""" An example to show how to use the extractor, handles regressing AMZN data """

from BalanceSheetDataExtractor import BalanceSheetDataExtractor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn import svm
import matplotlib.pyplot as plt
import math
from sklearn.metrics import mean_squared_error

def main():
    extractor = BalanceSheetDataExtractor('AMZN', '2010-12-31')
    data = extractor.get_all_data()

    #x = data[['CashAndCashEquivalentsAtCarryingValue', 'Assets', 'LiabilitiesCurrent', 'amznopen']]
    x = data[['amznopen']]
    #x = data[['amznopen']]
    y = data['amznclose']

    x_train, x_test = x[0:1000], x[1000:len(x)]
    y_train, y_test = y[0:1000], y[1000:len(y)]

    #clf = svm.SVR()
    #clf = DecisionTreeRegressor(max_depth=10)
    clf = LinearRegression()
    clf.fit(x_train, y_train)

    predictions = clf.predict(x_test)

    print predictions.shape
    print y_test.shape
    test_score = math.sqrt(mean_squared_error(y_test, predictions))
    train_score = math.sqrt(mean_squared_error(y_train, clf.predict(x_train)))
    print "Test score: " + str(test_score) + " RMSE error"
    print "Train score: " + str(train_score) + " RMSE error"
    #print clf.feature_importances_

    #print clf.score(x_train, y_train)
    #plt.plot(data.index, y, color='g')
    #plt.plot(data.index, clf.predict(x), color='r')
    #plt.plot(data.index, clf.predict(x), color='b')
    plt.plot(x_test.index, predictions, color='red')
    plt.plot(x_test.index, y_test, color='black')
    plt.show()

if __name__ == "__main__":
    main()
