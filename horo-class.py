#!/usr/local/bin/python
# Copyright (c) 2016 Allison Sliter
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the 'Software'), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED 'AS IS', WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

from __future__ import division
import random
import time
import pickle

from functools import partial
from collections import defaultdict
from nltk.tokenize.treebank import TreebankWordTokenizer
from nltk.stem.snowball import SnowballStemmer
from pandas import DataFrame
# you may need to install this (`pip install pandas`), or modify my
# code to use the built-in `csv` module --KG
from argparse import ArgumentParser

from perceptron import MulticlassAveragedPerceptron

# you'll need to write your own! --KG

# default number of epochs
EPOCHS = 20
stemmer = SnowballStemmer('english')

start = time.time()

#first pass features
elements = set(["air", "fire", "water", "earth"])
planets = set(["sun", "moon", "mercury", "venus", "mars", "earth", "jupiter", "saturn"])
meanings = set(["ram", "bull", "twins", "crab", "lion", "virgin", "scale", "scales", "scorpion", "archer", "sea-goat", "waterbearer", "fish"])
personality = set( ['Able', 'Accepting', 'Adventurous', 'Aggressive', 'Ambitious', 'Annoying', 'Arrogant', 'Articulate', 'Athletic', 'Awkward', 'Boastful', 'Bold', 'Bossy', 'Brave', 'Bright', 'Busy', 'Calm', 'Careful', 'Careless', 'Caring', 'Cautious', 'Cheerful', 'Clever', 'Clumsy', 'Compassionate', 'Complex', 'Conceited', 'Confident', 'Considerate', 'Cooperative', 'Courageous', 'Creative', 'Curious', 'Dainty', 'Daring', 'Dark', 'Defiant', 'Demanding', 'Determined', 'Devout', 'Disagreeable', 'Disgruntled', 'Dreamer', 'Eager', 'Efficient', 'Embarrassed', 'Energetic', 'Excited', 'Expert', 'Fair', 'Faithful', 'Fancy', 'Fighter', 'Forgiving', 'Free', 'Friendly', 'Friendly', 'Frustrated', 'Fun-loving', 'Funny', 'Generous', 'Gentle', 'Giving', 'Gorgeous', 'Gracious', 'Grouchy', 'Handsome', 'Happy', 'Hard-working', 'Helpful', 'Honest', 'Hopeful', 'Humble', 'Humorous', 'Imaginative', 'Impulsive', 'Independent', 'Intelligent', 'Inventive', 'Jealous', 'Joyful', 'Judgmental', 'Keen', 'Kind', 'Knowledgeable', 'Lazy', 'Leader', 'Light', 'Light-hearted', 'Likeable', 'Lively', 'Lovable', 'Loving', 'Loyal', 'Manipulative', 'Materialistic', 'Mature', 'Melancholy', 'Merry', 'Messy', 'Mischievous', 'Naive', 'Neat', 'Nervous', 'Noisy', 'Obnoxious', 'Opinionated', 'Organized', 'Outgoing', 'Passive', 'Patient', 'Patriotic', 'Perfectionist', 'Personable', 'Pitiful', 'Plain', 'Pleasant', 'Pleasing', 'Poor', 'Popular', 'Pretty', 'Prim', 'Proper', 'Proud', 'Questioning', 'Quiet', 'Radical', 'Realistic', 'Rebellious', 'Reflective', 'Relaxed', 'Reliable', 'Religious', 'Reserved', 'Respectful', 'Responsible', 'Reverent', 'Rich', 'Rigid', 'Rude', 'Sad', 'Sarcastic', 'Self-confident', 'Self-conscious', 'Selfish', 'Sensible', 'Sensitive', 'Serious', 'Short', 'Shy', 'Silly', 'Simple', 'Simple-minded', 'Smart', 'Stable', 'Strong', 'Stubborn', 'Studious', 'Successful', 'Tall', 'Tantalizing', 'Tender', 'Tense', 'Thoughtful', 'Thrilling', 'Timid', 'Tireless', 'Tolerant', 'Tough', 'Tricky', 'Trusting', 'Ugly', 'Understanding', 'Unhappy', 'Unique', 'Unlucky', 'Unselfish', 'Vain', 'Warm', 'Wild', 'Willing', 'Wise', 'Witty'] )
signs = set(["Aries, Taurus, Gemini, Cancer, Leo, Virgo, Libra, Scorpio, Sagittarius, Aquarius, Pisces, Capricorn"])
horoscope_words = set(["ascend", "ascending", "house", "rising", "celestial" ])





def extract_features(text):
    global stemmer
    global features
    text = text.upper()
    tokens = TreebankWordTokenizer().tokenize(text)
    stemmed = list()
    for t in tokens:
        stemmed.append(stemmer.stem(t))
        
    textfeatures = set()
    for t in stemmed:
        if features:
            norm = t.upper()
            norm = stemmer.stem(norm)
            textfeatures.add(norm)

    
    return frozenset(textfeatures)
    
    


if __name__ == "__main__":
    # command line arguments
    argparser = ArgumentParser(description="Zodiac sign classifier")
    argparser.add_argument("train", help="Training data CSV ")
    argparser.add_argument("test", help="Test data CSV")
    argparser.add_argument("-e", "--epochs", type=int, default=EPOCHS,
                           help="# of training epochs")
    args = argparser.parse_args()
    setlist = [(elements), (planets), (meanings), (personality), (signs), (horoscope_words), (elements | planets), (elements | meanings), (elements | personality), (elements | signs), (elements | horoscope_words), (planets | meanings), (planets | personality), (planets | signs), (planets | horoscope_words), (meanings | personality), (meanings | signs), (meanings | horoscope_words), (personality | signs), (personality | horoscope_words), (signs | horoscope_words), (elements | planets | meanings), (elements | planets | personality), (elements | planets | signs), (elements | planets | horoscope_words), (elements | meanings | personality), (elements | meanings | signs), (elements | meanings | horoscope_words), (elements | personality | signs), (elements | personality | horoscope_words), (elements | signs | horoscope_words), (planets | meanings | personality), (planets | meanings | signs), (planets | meanings | horoscope_words), (planets | personality | signs), (planets | personality | horoscope_words), (planets | signs | horoscope_words), (meanings | personality | signs), (meanings | personality | horoscope_words), (meanings | signs | horoscope_words), (personality | signs | horoscope_words), (elements | planets | meanings | personality), (elements | planets | meanings | signs), (elements | planets | meanings | horoscope_words), (elements | planets | personality | signs), (elements | planets | personality | horoscope_words), (elements | planets | signs | horoscope_words), (elements | meanings | personality | signs), (elements | meanings | personality | horoscope_words), (elements | meanings | signs | horoscope_words), (elements | personality | signs | horoscope_words), (planets | meanings | personality | signs), (planets | meanings | personality | horoscope_words), (planets | meanings | signs | horoscope_words), (planets | personality | signs | horoscope_words), (meanings | personality | signs | horoscope_words), (elements | planets | meanings | personality | signs), (elements | planets | meanings | personality | horoscope_words), (elements | planets | meanings | signs | horoscope_words), (elements | planets | personality | signs | horoscope_words), (elements | meanings | personality | signs | horoscope_words), (planets | meanings | personality | signs | horoscope_words), (elements | planets | meanings | personality | signs | horoscope_words)]
    setid = ["elements", "planets", "meanings", "personality", "signs", "horoscope_words", "elements planets", "elements meanings", "elements personality", "elements signs", "elements horoscope_words", "planets meanings", "planets personality", "planets signs", "planets horoscope_words", "meanings personality", "meanings signs", "meanings horoscope_words", "personality signs", "personality horoscope_words", "signs horoscope_words", "elements planets meanings", "elements planets personality", "elements planets signs", "elements planets horoscope_words", "elements meanings personality", "elements meanings signs", "elements meanings horoscope_words", "elements personality signs", "elements personality horoscope_words", "elements signs horoscope_words", "planets meanings personality", "planets meanings signs", "planets meanings horoscope_words", "planets personality signs", "planets personality horoscope_words", "planets signs horoscope_words", "meanings personality signs", "meanings personality horoscope_words", "meanings signs horoscope_words", "personality signs horoscope_words", "elements planets meanings personality", "elements planets meanings signs", "elements planets meanings horoscope_words", "elements planets personality signs", "elements planets personality horoscope_words", "elements planets signs horoscope_words", "elements meanings personality signs", "elements meanings personality horoscope_words", "elements meanings signs horoscope_words", "elements personality signs horoscope_words", "planets meanings personality signs", "planets meanings personality horoscope_words", "planets meanings signs horoscope_words", "planets personality signs horoscope_words", "meanings personality signs horoscope_words", "elements planets meanings personality signs", "elements planets meanings personality horoscope_words", "elements planets meanings signs horoscope_words", "elements planets personality signs horoscope_words", "elements meanings personality signs horoscope_words", "planets meanings personality signs horoscope_words", "elements planets meanings personality signs horoscope_words"]
    count = 1
    resultsdict = dict()
    for x in range(0, len(setlist)):
        forstart = time.time()
        features = setlist[x]
        # train
        print "Loading data, featureset", setid[x]
        count += 1
        source = DataFrame.from_csv(args.train)
        Phi_train = [extract_features(text) for text in source["text"]]
        Y_train = list(source.index)
        classifier = MulticlassAveragedPerceptron()
        classifier.fit(Y_train, Phi_train, epochs=args.epochs)
        # test
        n_correct = 0
        source = DataFrame.from_csv(args.test)
        for (y, row) in source.iterrows():
            phi = extract_features(row["text"])
            yhat = classifier.predict(phi)
            if y == yhat:
                n_correct += 1
        print "Accuracy: {:.4f}".format(n_correct / len(source))
        fortime  = time.time()
        resultsdict[setid[x]] = n_correct / len(source)
        print "this set iteration took", fortime-forstart, "seconds"
    
    with open("resultsdict.pkl", 'wb') as b:
        pickle.dump(resultsdict, b)
