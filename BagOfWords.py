#  This file contains code to accompany the Kaggle tutorial
#  "Deep learning goes to the movies".  The code in this file
#  is for Part 1 of the tutorial on Natural Language Processing.
#
# *************************************** #
#  Author: Angela Chapman

import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from KaggleWord2VecUtility import KaggleWord2VecUtility
import pandas as pd
import numpy as np

if __name__ == '__main__':
        train = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'labeledTrainData.tsv'), header = 0,\
                        delimiter = "\t", quoting = 3)
        test - pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'testData.tsv'), header = 0, \
                        delimiter = "\t", quoting = 3)
        print 'The first review is:'
        print train["review"][0]

        raw_input("Press Enter to continue...")

        print 'Download text data sets. If you already have NLTK datasets downloaded, just close the Python download window,,,'
        #nltk.download()        #Download text data sets, including stop words

        # Initialize an empty list to hold the clean reviews
        clean_train_reviews = []

        # Lopp over each review; create an index i that foes from 0 to the length of the movie review list

        print "Cleaning and parsing the training set movie reviews...\n"
        for i in xrange(0, len(train["review"])):
            clean_train_reviews.append(" ".join(KaggleWord2VecUtility.review_to_wordlist(train["review"][i], True)))

        #***** Create a bag of words from the training sets
        print "Creating the bag of words...\n"

        #Initialize the "CountVectorizer" object, which is scikit-learn's bag of words tool
        vectorizer = CounVectorizer(analyzer = "word", \
                                    tokenizer = None, \
                                    preprocessor = None, \
                                    stop_words = None, \
                                    max_features = 5000)
        #fit_transform() does two functions: first, it fits the model and learns the vocabulary;
        #second, it transforms our training data into feature vectors. The input to fit_transform should be a list of strings.
        #
        #This may take a few minutes to run
        forest = forest.fit(train_data_features, train["sentiment"])

        #creat an empty list and append the clean reviews one by one
        clean_test_reviews = []

        print "Cleaning and parsing the test set movie reviews...\n"
        for i in xrange(0, len(test["review"])):
            clean_test_reviews.append(" ".join(KaggleWord2VecUtility.review_to_wordlist(test["review"][i], True)))

            # Get a bag of words for the test set, and convert to numpy array
            test_data_features = vectorizer.transform(clean_test_reviews)
            np.asarray(test_data_features)

            # use the random forest to make sentiment label predictions
            print "Predicting test labels...\n"
            result = forest.predict(test_data_features)

            # Copy the results to a pandas dataframe with an "id" column and a "sentiment" column
            output = pd.dataframe(data = {"id": test["id"], "sentiment":result})

            # Use pandas to write the comma-separated output file
            output.to_csv(os.path.join(os.path.dirname(__file__), 'data', 'Bag_of_Words_model.csv'), index = False, quoting = 3)
            print "Wrote resuls to Bag_of_words_model.csv"
