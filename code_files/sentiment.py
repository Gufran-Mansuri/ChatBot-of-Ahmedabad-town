""" An example of sentiment analysis

Sam Scott, Mohawk College, 2021
"""
### Load docs and labels
filenames = ["amazon_cells_labelled.txt", "imdb_labelled.txt", "yelp_labelled.txt"]
docs = []
labels = []
for filename in filenames:
    with open("sentiment/"+filename) as file:
        for line in file:
            line = line.strip()
            labels.append(int(line[-1]))
            docs.append(line[:-2].strip())
result = []

for i in range(50):
    ## split into training and testing data
    from sklearn.model_selection import train_test_split

    split = train_test_split(docs, labels)
    train_docs, test_docs, train_labels, test_labels = split

    ## Vectorize
    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer = CountVectorizer()
    train_data = vectorizer.fit_transform(train_docs)
    test_data = vectorizer.transform(test_docs)


    ## create and train the classifier
    from sklearn.naive_bayes import ComplementNB

    clf = ComplementNB()
    clf.fit(train_data, train_labels)


    ## get predictions for unseen examples
    pred = clf.predict(test_data)


    ## report accuracy
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import confusion_matrix
    accuracy = accuracy_score(test_labels, pred)
    result.append(accuracy)
    print("Accuracy: ",accuracy)

avg = sum(result)/len(result)*100
print("avg Accuracy score: ", avg)
#print("Confusion Matrix:")
#print(confusion_matrix(test_labels, pred))

## Make a production batch
vectorizer = CountVectorizer()
vectors = vectorizer.fit_transform(docs)
clf = ComplementNB()
clf.fit(vectors, labels)

from joblib import dump
dump(clf, "toneClassifier.joblib")
dump(vectorizer, "toneVectorizer.joblib")
