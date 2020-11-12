# Natural Language Processing

# Importing the libraries

import numpy as np
import pandas as pd
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestRegressor
import statsmodels.formula.api as sm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from xgboost import XGBClassifier


def nltk():
    # Importing the dataset
    dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter='\t', quoting=3)
    corpus = []
    unique_words = set()
    for index, row in dataset.iterrows():
        review = re.sub('[^a-zA-Z]', ' ', row['Review'])
        review = review.lower()
        review = review.split()
        ps = PorterStemmer()
        review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
        unique_sentence_words = set(review)
        review = ' '.join(review)
        unique_words |= unique_sentence_words
        corpus.append(review)

    # Creating the Bag of Words model

    cv = CountVectorizer(max_features = len(unique_words))

    # tf = HashingVectorizer(n_features = 1565)
    X = cv.fit_transform(corpus).toarray()
    y = dataset.iloc[:, 1].values

    '''

    # Building the optimal model using Backward Elimination

    max_value = 1

    col_list = [i for i in range(len(unique_words))]

    max_p_col = -1
    count = 1

    while(max_value > 0.05):

        if max_p_col != -1:
            del col_list[max_p_col]


        X_Opt = X[:, col_list]
        try:
            regressor_OLS = sm.OLS(endog=y, exog=X_Opt).fit()
        except:
            df = pd.DataFrame(data=X_Opt[1:, 1:],
                              index = X_Opt[1:, 0],
                              columns = X_Opt[0, 1:])
            df.to_csv('input_values.csv')
            break
        p_values = list(regressor_OLS.pvalues)
        max_value = max(p_values)
        max_p_col = p_values.index(max_value)

        print('max_p_value and its columns are',max_value, col_list[max_p_col])
        count = count+1

        '''

    # Splitting the dataset into the Training set and Test set

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


    # Fitting Logistic Regression to the Training set


    '''
    classifier = LogisticRegression(random_state=0) # Accuracy is 0.71
    
    '''

    # Fitting XG Boost to the Training set

    
    classifier = XGBClassifier()
    


    # Fitting Naive Bayes to the Training set

    '''
    classifier = GaussianNB() # Accuracy is 0.73
    '''
    
    # Fitting K-NN to the Training set

    
    # classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2) # Accuracy is 0.61


    # Fitting SVM to the Training set

    '''
    
    classifier = SVC(kernel='linear', random_state=0) # Accuracy is 0.72
    '''

    '''

    # Fitting RandomForestClassification to the Training set
    classifier = RandomForestClassifier(n_estimators=5, criterion='entropy', random_state=0)
    '''
    classifier.fit(X_train, y_train)

    # Predicting the Test set results
    y_pred = classifier.predict(X_test)

    # Accuracy score
    # accuracy = accuracy_score(y_test, y_pred)

    # print('Model Accuracy = {},number of columns ={}'.format(accuracy, 1565))

    # Making the Confusion Matrix
    accuracy = accuracy_score(y_test, y_pred)

    cm = confusion_matrix(y_test, y_pred)


    print('accuracy value is==',accuracy)
    return cm


if __name__=="__main__":

    cm = nltk()

    print('confusioin matrix', cm)

    # data_format()
