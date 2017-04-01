""" An example to show how to use the extractor, handles regressing AMZN data """

from BalanceSheetDataExtractor import BalanceSheetDataExtractor
from sklearn.tree import DecisionTreeRegressor
from sklearn import svm
import matplotlib.pyplot as plt

def main():
    extractor = BalanceSheetDataExtractor('AMZN', '2010-12-31')
    data = extractor.get_all_data()

    x = data[['CashAndCashEquivalentsAtCarryingValue', 'Assets', 'LiabilitiesCurrent']]
    y = data['amznclose']

    x_train, x_test = x[0:1400], x[1400:len(x)]
    y_train, y_test = y[0:1400], y[1400:len(y)]

    #clf = svm.SVR()
    clf = DecisionTreeRegressor(max_depth=10)
    clf.fit(x, y)
    #print clf.score(x_train, y_train)
    plt.plot(data.index, y)
    plt.plot(data.index, clf.predict(x))
    plt.show()

if __name__ == "__main__":
    main()
