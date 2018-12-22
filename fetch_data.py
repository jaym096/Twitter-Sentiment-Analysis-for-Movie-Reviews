import json
import tweepy

def fetchMovieData(name):
    #configuration file
    data = open("Data/config.json")
    config = json.load(data)
    
    # Configure the OAuth using the Credentials provided
    consumer_key= str(config["consumer_key"])
    consumer_secret= str(config["consumer_secret"])
    
    access_token=str(config["access_token"])
    access_token_secret=str(config["access_token_secret"])
    
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    myApi = tweepy.API(auth)
    
    #Open a file to write data
    txtFile = name+".txt"
    print "file name: ",txtFile
    fw = open(txtFile,'w')
    
    #list of unique users who tweeted
    users = []
    
    max_twt = 1500
    count= 1
    # fetch the Tweets based on the query
    for tweet in tweepy.Cursor(myApi.search, q=name, result_type="recent", tweet_mode='extended', include_entities=True, lang="en").items(max_twt):
        string = ""    
        followers_count = tweet.author._json['followers_count']
        screen_name = tweet.author._json['screen_name']
        twt = tweet.full_text
        #For every tweet that is fetched, write only relevant tweets    
        if(followers_count > 20 and screen_name not in users and not twt.startswith("RT")):
            string = str(count) + " "
            string = string + " ".join(twt.split("\n")).encode(encoding='utf-8') + "\n"
            fw.write(string)
            count += 1
    fw.close()  