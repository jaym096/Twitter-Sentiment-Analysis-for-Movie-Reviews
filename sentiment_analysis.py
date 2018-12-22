import sys
import re
import os
import nltk
import fetch_data
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from textblob import TextBlob
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from nltk.tokenize import RegexpTokenizer


# Variable Initialization
vNegative = []
Negative = []
Positive = []
vPositive = []
data_X = ""
data_Y = ""


def generateStopWordsList():
    #Fetch the Text File which has all the stopwords from the PATH
    File = "Data/stopwords.txt"
    
    #stopwords list    
    stopwords_list = []

    #Open the stopwords file read the data and store in a list
    try:
        fp = open(File, 'r')
        line = fp.readline()
        while line:
            word = line.strip()
            stopwords_list.append(word)
            line = fp.readline()
        fp.close()
    except:
        print("ERROR: Opening File")
    return stopwords_list
    
    
    
def generateAffinintyList(dataPath):
    affin_dataset = "Data/affin_data.txt"
    try:
        affin_list = open(affin_dataset).readlines()
    except:
        print("ERROR: Opening File", affin_dataset)
        exit(0)
    
    return affin_list
    
    

def createDictionary(affin_list):
    # Create list to store the words and its score i.e. polarity
    words = []
    score = []

    # for every word in AFF-111 list, generate the Words with their scores (polarity)
    for word in affin_list:
        words.append(word.split("\t")[0].lower())
        score.append(int(word.split("\t")[1].split("\n")[0]))

    #Categorize words into different Categories
    for i in range(len(words)):
        if score[i] == -4 or score[i] == -5:
            vNegative.append(words[i])
        elif score[i] == -3 or score[i] == -2 or score[i] == -1:
            Negative.append(words[i])
        elif score[i] == 3 or score[i] == 2 or score[i] == 1:
            Positive.append(words[i])
        elif score[i] == 4 or score[i] == 5:
            vPositive.append(words[i])
            
"""
    This function is used for preprocessing the data.
    Here we clean the data, do dimensionaltiy reduction steps
"""
def preprocessing(dataSet):

    processed_data = []

    #Make a list of all the Stopwords to be removed
    stopWords = generateStopWordsList()
    #For every TWEET in the dataset do,
    for tweet in dataSet:
        temp_tweet = tweet
        
        #Remove Emojis
        emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)
        tweet = emoji_pattern.sub(r'', tweet) # no emoji

        #remove @username
        tweet = re.sub('@[^\s]+','',tweet).lower()

        #Replace all Punctuations except hashtag
        tweet = tweet.replace("[^a-zA-Z#]", " ")
               
        #Replace all the numeric terms
        tweet = re.sub('[0-9]+', "",tweet)
        
        #Remove urls
        tweet = re.sub('http.?://[^\s]+[\s]?', "", tweet)

        #Remove all the STOP WORDS
        for sw in stopWords:
            if sw in tweet:
                tweet = re.sub(r'\b' + sw + r'\b'+" ","",tweet)
                
        #Remove the unnecessary white spaces
        tweet = re.sub('[\s]+',' ', tweet)

        tweet.replace(temp_tweet, tweet)

        #Save the Processed Tweet after data cleansing
        processed_data.append(tweet)
    return processed_data


"""
    This function is used to generate the Feature Vectors for the Training Data,
    and assign a class label to it accordingly
"""
def FeaturizeTrainingData(dataset, type_class):

    neutral_list = []
    i=0

    # For each Tweet split the Tweet by " " (space) i.e. split every word of the Tweet
    data = [tweet.strip().split(" ") for tweet in dataset]
    #print(data)

    # Feature Vector is to store the feature of the TWEETs
    feature_vector = []

    # for every sentence i.e. TWEET find the words and their category
    for sentence in data:
        # Category count for every Sentence or TWEET
        vNegative_count = 0
        Negative_count = 0
        Positive_count = 0
        vPositive_count = 0

        # for every word in sentence, categorize
        # and increment the count by 1 if found
        for word in sentence:
            if word in vPositive:
                vPositive_count = vPositive_count + 1
            elif word in Positive:
                Positive_count = Positive_count + 1
            elif word in vNegative:
                vNegative_count = vNegative_count + 1
            elif word in Negative:
                Negative_count = Negative_count + 1
        i+=1

        #Assign Class Label
        if vPositive_count == vNegative_count == Positive_count == Negative_count:
            feature_vector.append([vPositive_count, Positive_count, Negative_count, vNegative_count, "neutral"])
            neutral_list.append(i)
        else:
            feature_vector.append([vPositive_count, Positive_count, Negative_count, vNegative_count, type_class])
    #print(neutral_list)
    return feature_vector

    
"""
    This function is used to generate the Feature Vectors for the Test Data
"""
def FeatureizeTestData(dataset):
    data = [tweet.strip().split(" ") for tweet in dataset]
    #print(data)
    #count_Matrix = []
    feature_vector = []

    for sentence in data:
        #print(word)
        vNegative_count = 0
        Negative_count = 0
        Positive_count = 0
        vPositive_count = 0

        # for every word in sentence, categorize
        # and increment the count by 1 if found
        for word in sentence:
            if word in vPositive:
                vPositive_count = vPositive_count + 1
            elif word in Positive:
                Positive_count = Positive_count + 1
            elif word in vNegative:
                vNegative_count = vNegative_count + 1
            elif word in Negative:
                Negative_count = Negative_count + 1

        if (vPositive_count + Positive_count) > (vNegative_count + Negative_count):
            feature_vector.append([vPositive_count, Positive_count, Negative_count, vNegative_count, "positive"])
            #neutral_list.append(i)
        elif (vPositive_count + Positive_count) < (vNegative_count + Negative_count):
            feature_vector.append([vPositive_count, Positive_count, Negative_count, vNegative_count, "negative"])
        else:
            feature_vector.append([vPositive_count, Positive_count, Negative_count, vNegative_count, "neutral"])

        #count_Matrix.append([vPositive_count, Positive_count, Negative_count, vNegative_count])
    return feature_vector
    

"""
    This function is used to classify the Data using
    Gaussian Naive Bayes Algorithm
"""
def classify_naive_bayes(train_X, train_Y, test_X):
    print("Classifying using Gaussian Naive Bayes ...")
    gnb = GaussianNB()
    yHat = gnb.fit(train_X,train_Y).predict(test_X)
    return yHat


"""
    This function is used to classify the Data using
    Support Vector Machine Algorithm
"""
def classify_svm(train_X, train_Y, test_X):
    print("Classifying using Support Vector Machine ...")
    clf = SVC()
    clf.fit(train_X,train_Y)
    yHat = clf.predict(test_X)
    return yHat
    

#########FOR TEST DATA CLASSIFICATION########
def hashtag_extract(x):
    hashtags = []
    # Loop over the words in the tweet
    for i in x:
        ht = re.findall(r"#(\w+)", i)
        hashtags.append(ht)
    return hashtags

def provide_results(yHat, DA, test_data2):
    pCount = 0  #positive
    nCount = 0  #negative
    NCount = 0  #neutral
    temp_yHat = yHat
    for r in temp_yHat:
        if r == "positive":
            pCount += 1
        if r == "negative":
            nCount += 1
        if r == "neutral":
            NCount += 1
    print "POSITIVE : ", pCount
    print "NEGATIVE : ", nCount
    print "NEUTRAL : ", NCount
    
    #Display analysis
    if DA == "bar" or DA == "Bar":
        HT_negative = []
        HT_regular = []
        for tweets in test_data2:
            tokenizer = RegexpTokenizer(r'\w+')
            token_words = tokenizer.tokenize(tweets)
            for word in token_words:
                blob = TextBlob(word).sentiment
                if blob.polarity < 0:
                    HT_negative.append(word)
                if blob.polarity > 0:
                    HT_regular.append(word)
        a = nltk.FreqDist(HT_regular)
        d = pd.DataFrame({'Hashtag': list(a.keys()),'Count': list(a.values())})
        
        # selecting top 10 most frequent hashtags     
        d = d.nlargest(columns="Count", n = 10)
        plt.figure(figsize=(16,5))
        ax = sns.barplot(data=d, x="Hashtag", y="Count")
        ax.set(ylabel = 'Count')
        plt.show()
    if DA == "WC" or DA == "Wc" or DA == "wc":
        all_words = ' '.join([text for text in test_data2])
        wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words)        
        plt.figure(figsize=(10, 7))        
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis('off')
        plt.show()
    if DA == "Pie" or DA == "pie":
        labels = 'Positive', 'Negative', 'Neutral'
        sizes = [pCount, nCount, NCount]
        colors = ['lightskyblue', 'red', 'yellowgreen']
        # Plot
        plt.pie(sizes, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=140)
        plt.axis('equal')
        plt.show()


def classify_naive_bayes_twitter(train_X, train_Y, test_X, test_Y, DA, test_data2):
    print test_Y
    print train_Y
    print("Classifying using Gaussian Naive Bayes ...")
    gnb = GaussianNB()
    yHat = gnb.fit(train_X,train_Y).predict(test_X)
    print "Prediction from gnb..."
    provide_results(yHat, DA, test_data2)


def classify_svm_twitter(train_X, train_Y, test_X, test_Y, DA, test_data2):
    print("Classifying using Support Vector Machine ...")
    clf = SVC()
    clf.fit(train_X,train_Y)
    yHat = clf.predict(test_X)
    print "Prediction from SVM..."
    provide_results(yHat, DA, test_data2)
    accuracy = accuracy_score(test_Y, yHat)
    print "Accuracy : ", "%.3f"%accuracy
    
    
def classify_twitter_data(DA, file_name):
    print "directory:", dirPath
    test_data = open(dirPath+"/twitter sentiment analysis/"+file_name).readlines()
    test_data = preprocessing(test_data)
    test_data2 = test_data
    test_data = FeatureizeTestData(test_data)
    test_data = np.reshape(np.asarray(test_data),newshape=(len(test_data),5))
    
    #Split Data into Features and Classes
    data_X_test = test_data[:,:4].astype(int)
    data_Y_test = test_data[:,4]

    print("Classifying...")
    #Classify
    if sys.argv[2] == "gnb":
        classify_naive_bayes_twitter(data_X, data_Y, data_X_test, data_Y_test, DA, test_data2)
    elif sys.argv[2] == "svm":
        classify_svm_twitter(data_X, data_Y, data_X_test, data_Y_test, DA, test_data2)


# main
if __name__ == "__main__":
    
    name = sys.argv[1]
    print "name: ", name
    
    print("please wait, fetching tweets for the movie...")
    movieData = fetch_data.fetchMovieData(str(name))
    
    #fetch th current working dir
    os.chdir('../')        #!!!!!IMPORTANT UNCOMMENT
    dirPath = os.getcwd()
    
    # STEP 1: Generate Affinity List
    print("Please wait while we Classify your data ...")
    affin_list = generateAffinintyList(dirPath+"/Data/Affin_Data.txt")
    
    # STEP 2: Create Dictionary based on Polarities from the Lexicons
    createDictionary(affin_list)
    
    # STEP 3: Read Data positive and negative Tweets, and do PREPROCESSING
    print("Reading your data ...")
    positive_data = open(dirPath+"/twitter sentiment analysis/Data/rt-polarity-pos.txt").readlines()
    print("Preprocessing in progress ...")
    positive_data = preprocessing(positive_data)
    #print(positive_data)

    negative_data = open(dirPath+"/twitter sentiment analysis/Data/rt-polarity-neg.txt").readlines()
    negative_data = preprocessing(negative_data)
    #print(negative_data)
    
    # STEP 4: Create Feature Vectors and Assign Class Label for Training Data
    print("Generating the Feature Vectors ...")
    positive_sentiment = FeaturizeTrainingData(positive_data, "positive")
    negative_sentiment = FeaturizeTrainingData(negative_data,"negative")
    final_data = positive_sentiment + negative_sentiment
    final_data = np.reshape(np.asarray(final_data),newshape=(len(final_data),5))
    
    #Split Data into Features and Classes
    data_X = final_data[:,:4].astype(int)   
    data_Y = final_data[:,4]
    
    DA = sys.argv[3]
    classify_twitter_data(DA, file_name=name+".txt")