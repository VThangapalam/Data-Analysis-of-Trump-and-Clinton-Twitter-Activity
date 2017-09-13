import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold
import numpy as np
import re
import conf

#import sys
#sys.stdout = open("classify_log.txt", "w")

def train_test_split(tweets):
    """DONE.
    Returns a random split of the ratings matrix into a training and testing set.
    """
    test = set(range(len(tweets))[::10])
    train = sorted(set(range(len(tweets))) - test)
    test = sorted(test)
    return tweets.iloc[train], tweets.iloc[test]



def getDataLabel(dataSet):    
    labels=[]
    data=[]
    for index,row in dataSet.iterrows():
        labels.append(row['SENTIMENT'])
        data.append(row['DATA'])
    return data,labels
        
     
     
def tokenize(string):
    """Given a string, return a list of tokens such that: (1) all
    tokens are lowercase, (2) all punctuation is removed. Note that
    underscore (_) is not considered punctuation.
    Params:
        text....a string
    Returns:
        a list of tokens
    """
    ###TODO
   
    string = re.sub('http\S+', 'THIS_IS_A_URL', string)
    return re.findall('\w+', string.lower())  
     
def featurizeData(dataTrain):
 
    """vectorizer = CountVectorizer(min_df=1, ngram_range=(1,1))
    X = vectorizer.fit_transform(dataTrain)
    print('vectorized %d tweets. found %d terms.' % (X.shape[0], X.shape[1]))
    
    model = LogisticRegression()
    model.fit(X, labels)"""
    vec = CountVectorizer(input='content', tokenizer=tokenize,
                          binary=True, min_df=1, max_df=.8,
                          ngram_range=(1,1))
   
    X = vec.fit_transform(dataTrain)
    vocab = np.array(vec.get_feature_names())
    return (X, vocab)
    
    
def featurizeDataTest(dataTest,vocab):
 
    """vectorizer = CountVectorizer(min_df=1, ngram_ran
    ge=(1,1))
    X = vectorizer.fit_transform(dataTrain)
    print('vectorized %d tweets. found %d terms.' % (X.shape[0], X.shape[1]))
    
    model = LogisticRegression()
    model.fit(X, labels)"""
    vec = CountVectorizer(input='content', tokenizer=tokenize,vocabulary =vocab,
                          binary=True, min_df=1, max_df=.8,
                          ngram_range=(1,1))
   
    X = vec.fit_transform(dataTest)
    return X

   
   
def fitmodel(X,labels):      
    model = LogisticRegression()
    model.fit(X, labels)   
    return model
    
    
    

    
    
def cross_validation_accuracy(clf, X, labels, k):
    """
    Compute the average testing accuracy over k folds of cross-validation. You
    can use sklearn's KFold class here (no random seed, and no shuffling
    needed).
    Params:
      clf......A LogisticRegression classifier.
      X........A csr_matrix of features.
      labels...The true labels for each instance in X
      k........The number of cross-validation folds.
    Returns:
      The average testing accuracy of the classifier
      over each fold of cross-validation.
    """
    ###TODO
    cv = KFold(len(labels), k)
    accuracies = []
    for train_ind, test_ind in cv:
        clf.fit(X[train_ind], labels[train_ind])
        predictions = clf.predict(X[test_ind])
        accuracies.append(accuracy_score(labels[test_ind], predictions))
    return np.mean(accuracies)
   
def accuracy_score(truth, predicted):
    """ Compute accuracy of predictions.
    DONE ALREADY
    Params:
      truth.......array of true labels (0 or 1)
      predicted...array of predicted labels (0 or 1)
    """
    return len(np.where(truth==predicted)[0]) / len(truth)
       
# Compute accuracy
def accuracy(truth, predicted):
    j=0
    correct=0
    for i in range(0,len(truth)):
        if(truth[i]==predicted[i]):
            correct=correct+1
      
    return correct / len(truth)   
   
def getLabelName(val):
    if(val==0):
        return 'negative'
        
    if(val==4):
        return 'positive'
        
    if(val==2):
        return 'neutral' 
    else :
        return 'undefined'
        
        
        
def top_coefs(clf, label, n, vocab):
    """
    Find the n features with the highest coefficients in
    this classifier for this label.
    See the .coef_ attribute of LogisticRegression.
    Params:
      clf.....LogisticRegression classifier
      label...1 or 0; if 1, return the top coefficients
              for the positive class; else for negative.
      n.......The number of coefficients to return.
      vocab...Dict from feature name to column index.
    Returns:
      List of (feature_name, coefficient) tuples, SORTED
      in descending order of the coefficient for the
      given class label.
    """
    ###TODO
    coef = clf.coef_[0]
    if(label==0):    
        top_coef_ind = np.argsort(coef)[::1][:n]
    else:
        top_coef_ind = np.argsort(coef)[::-1][:n]
    

    vocabdict ={}
    for cnt, v in enumerate(vocab):
        vocabdict[v]=cnt
    vocab_terms=list(vocabdict.keys())
    top_coef_terms =[]
    top_coef=[]
    for val in top_coef_ind:
        top_coef.append(abs(coef[val]))
        for key,value in vocabdict.items():
            if(value==val):
               top_coef_terms.append(key)
               break;
   
    res=[x for x in zip(top_coef_terms, top_coef)]
    return res        
        
    
def main():
    dataCol= pd.read_csv(conf.train_data_file)
    newTestData=pd.read_csv(conf.last_100_tweets_file)
    #print(dataCol.iloc[0])
    for index,row in dataCol.iterrows():           
        positiveTweets=dataCol[dataCol.SENTIMENT==4]
        negativetiveTweets=dataCol[dataCol.SENTIMENT==0]
        neutralTweets=dataCol[dataCol.SENTIMENT==2]
    str1=('\nTotal Number of tweets used for training : %d'%(len(dataCol)))
    str2=('Number of Positive tweets used for training : %d'%(len(positiveTweets)))
    str3=('Number of Negative tweets used for training : %d'%(len(negativetiveTweets)))
    str4=('Number of Neutral tweets used for training : %d'%(len(neutralTweets)))
    ratings_train, ratings_test = train_test_split(dataCol)
    str5=('\n%d training tweets; %d testing tweets' % (len(ratings_train), len(ratings_test)))
    traindata,trainlabel = getDataLabel(ratings_train)
    testdata,testlabel = getDataLabel(ratings_test)
    X,vocab = featurizeData(traindata)
    model = fitmodel(X,trainlabel)
    test_matrix= featurizeDataTest(testdata,vocab)
    predicted = model.predict(test_matrix)
    np_predicted = np.array(predicted)
    negative_predicted_label_indices=np.where(np_predicted == 0)
    positive_predicted_label_indices=np.where(np_predicted == 4)
    neutral_predicted_label_indices=np.where(np_predicted == 2)
    str5=('\nSample of a test tweet predicted as negative:')
    str6=(testdata[negative_predicted_label_indices[0][0]])
    str7=('Actual label : %s'%(getLabelName(testlabel[negative_predicted_label_indices[0][0]]))+'\n')
    str8=('Sample of a test tweet predicted as positive')
    str9=(testdata[positive_predicted_label_indices[0][0]])
    
    str41=('Actual label : %s'%(getLabelName(testlabel[positive_predicted_label_indices[0][0]]))+'\n')
    
    str10=('Sample of a test tweet predicted as neutral')
    str11=(testdata[neutral_predicted_label_indices[0][0]])
    str12=('Actual label : %s'%(getLabelName(testlabel[neutral_predicted_label_indices[0][0]]))+'\n')
    str13=('accuracy on test data=%.3f' % accuracy(testlabel, predicted)+'\n')
    
    alldata,alllabel = getDataLabel(dataCol)
    all_matrix= featurizeDataTest(alldata,vocab)
    mean_accuracy = cross_validation_accuracy(model, all_matrix, np.array(alllabel), 5)
    str14=('Mean accuracy for 5 fold cross validation is : %f '%(mean_accuracy)+'\n')
    
    
        
     # Print top coefficients per class.
    str15=('\nTOP COEFFICIENTS PER CLASS:')
    str16=('negative words:')
    str17=('\n'.join(['%s: %.5f' % (t,v) for t,v in top_coefs(model, 1, 5, vocab)]))
    str18=('\npositive words:')
    str19=('\n'.join(['%s: %.5f' % (t,v) for t,v in top_coefs(model, 0, 5, vocab)])+'\n')
  
    
    str20=('Testing on new data collected')
    testdata_new,testlabel_default = getDataLabel(newTestData)
    new_test_matrix= featurizeDataTest(testdata_new,vocab)
    predicted_test = model.predict(new_test_matrix)
    np_predicted_test = np.array(predicted_test)
    negative_predicted_label_indices_test=np.where(np_predicted_test == 0)
    positive_predicted_label_indices_test=np.where(np_predicted_test == 4)
    neutral_predicted_label_indices_test=np.where(np_predicted_test == 2)
   
    if(len(negative_predicted_label_indices_test[0])>0):
        str21=('\nSample of a newly collected tweet predicted as negative:')
        str22=(newTestData.iloc[negative_predicted_label_indices_test[0][0]][1]+'\n')
    else :
        print('No new tweets collected predicted as negative')
   
    
    if(len(positive_predicted_label_indices_test[0])>0):
        str23=('Sample of a newly collected tweet predicted as positive')
        str24=(newTestData.iloc[positive_predicted_label_indices_test[0][0]][1]+'\n')
    else :
        print('No new tweets collected predicted as positive')
   
    if(len(neutral_predicted_label_indices_test[0])>0):
        str25=('Sample of newly collected tweet predicted as neutral')
        str26=(newTestData.iloc[neutral_predicted_label_indices[0][0]][1]+'\n')
    else :
        print('No new tweets collected predicted as neutral')
   
  
    str27=('Number of new tweets classified as negative %d '%(len(negative_predicted_label_indices_test[0])))
    str28=('Number of new tweets classified as positive %d '%(len(positive_predicted_label_indices_test[0])))
    str29 = ('Number of new tweets classified as neutral %d '%(len(neutral_predicted_label_indices_test[0])))

    
    f = open(conf.classify_log,'w')
   
    f.write(str1+'\n'+str2+'\n'+str3+'\n'+str4+'\n'+str5+'\n'+str6+'\n'+str7+'\n'+str8+'\n'+str9+'\n'+str41+'\n')
    f.write('\n')
    f.write(str10+'\n'+str11+'\n'+str12+'\n'+str13+'\n'+str14+'\n'+str15+'\n'+str16+'\n'+str17+'\n'+str18+'\n'+str19+'\n'+str20+'\n')
    f.write(str21+'\n'+str22+'\n'+str23+'\n'+str24+'\n'+str25+'\n'+str26+'\n'+str27+'\n'+str28+'\n'+str29+'\n')
    f.close()
   
    
    
if __name__ == '__main__':
    main()