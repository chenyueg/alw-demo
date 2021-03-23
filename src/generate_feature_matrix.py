import string

# File paths
tweetsFile = "../raw_data/retrieved_tweets_trial.txt"
wordsFile = "../features/abusive_words.txt"

def process_tweets(tweets):

    cleanTweetsWithLabels = []

    for each in tweets:
        tweet = each.strip("\n").split(" ")
        number = tweet.pop(0)
        tweetID = tweet.pop(0)
        label = tweet.pop()
        tweet = [word.strip(string.punctuation).lower() for word in tweet]
        # print([number, tweetID, tweet, label])
        cleanTweetsWithLabels.append([number, tweetID, tweet, label])

    return cleanTweetsWithLabels

def process_words(words):

    wordList = []

    for word in words:
        wordList.append(word.strip().lower())

    return wordList

def generate_matrix(tweets, words):

    wordsLentgh = len(words)
    featureMatrix = []

    for tweet in tweets:
        featureVector = [0] * wordsLentgh
        for i in range(wordsLentgh):
            # tweet[2] is the sub-list that contains the actual tweet
            featureVector[i] = tweet[2].count(words[i])
        featureMatrix.append(featureVector)

    return featureMatrix

with open(tweetsFile, encoding = "utf-8") as tweetsHandle:
    tweets = tweetsHandle.readlines()
with open(wordsFile, encoding = "utf-8") as wordsHandle:
    words = wordsHandle.readlines()

cleanTweetsWithLabels = process_tweets(tweets)
# print(cleanTweetsWithLabels)
wordList = process_words(words)
# print(wordList)
featureMatrix = generate_matrix(cleanTweetsWithLabels, wordList)
# print(len(featureMatrix), len(featureMatrix[0]))
# print(featureMatrix)