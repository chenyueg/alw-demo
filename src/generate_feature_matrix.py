import string

# File paths
tweetsFile = "../raw_data/retrieved_tweets.txt"
wordsFile = "../raw_data/abusive_words.txt"
timblTrainFile = "../features/timbl_feature_matrix_train"
timblTestFile = "../features/timbl_feature_matrix_test"

def extract_manual_features(tweet):

    '''
    Takes a not-yet-processed tweet in the form of [word1, word2, ..., wordn]
    returns a list of manual features adapted from ../resource/2872427.2883062.pdf

    - length of comment in tokens
    - average length of word
    - number of periods, question marks, quotes, and exclamation marks
    - number of one letter tokens
    - number of capitalized letters
    - number of non-alpha characters
    '''

    # print(tweet)
    length = len(tweet)
    averageWordLength = sum([len(word) for word in tweet])//length
    oneLetter = sum([1 for word in tweet if len(word) == 1])

    tweet = " ".join(tweet)

    punctuationCount = 0
    for each in [".","?","\"","\'","!"]:
        punctuationCount += tweet.count(each)

    capitalized = 0
    nonAlpha = 0
    for each in tweet:
        if each.isupper():
            capitalized += 1
        if not each.isalpha() and each != " ":
            nonAlpha += 1

    manual_features = [length, averageWordLength, oneLetter, punctuationCount, capitalized, nonAlpha]
    # print(manual_features)

    return manual_features

def process_tweets(tweets):

    '''
    Takes a list of retrieved tweets like ../raw_data/retrieved_tweets.txt
    Returns a list of pre-processed tweets in the format of [number, tweetID, tweet, manual, label]
    '''

    cleanTweetsWithLabels = []

    for each in tweets:
        tweet = each.strip("\n").split(" ")
        number = tweet.pop(0)
        tweetID = tweet.pop(0)
        label = tweet.pop()
        manual = extract_manual_features(tweet)
        tweet = [word.strip(string.punctuation).lower() for word in tweet]
        # print([number, tweetID, tweet, label])
        cleanTweetsWithLabels.append([number, tweetID, tweet, manual,label])

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
        # tweet[3] are the manual features
        featureVector.extend(tweet[3])
        # tweet[4] is the label
        featureVector.append(tweet[4])
        # print(tweet[3])
        featureMatrix.append(featureVector)

    # print(featureMatrix)

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

timblTrainHandle = open(timblTrainFile, "w+", encoding = "utf-8")
timblTestHandle = open(timblTestFile, "w+", encoding = "utf-8")

count = 1

for eachTweet in featureMatrix:

    # Change 9 to 5 to create a 20% test set
    if count % 9:
        timblTrainHandle.write(",".join([str(value) for value in eachTweet]))
        timblTrainHandle.write("\n")
    else:
        timblTestHandle.write(",".join([str(value) for value in eachTweet]))
        timblTestHandle.write("\n")

    count += 1

timblTrainHandle.close()
timblTestHandle.close()