import numpy
import urllib.request
import scipy.optimize
import random
from collections import defaultdict
import nltk
import string
from nltk.stem.porter import *
from sklearn import linear_model
from nltk.corpus import stopwords

class KeywordCount:
    def __init__(self, fname):
        self.name = fname  #file name of input transcript
        #init for updateWordCount()
        self.words = []  #sequenced word list
        self.wordCount = defaultdict(int)  #dictionary of words and their count
        self.wordCountSort = defaultdict(int)  #sorted dictionary
        self.transSize = 0  #words the transcript have
        self.transSizeNS = 0  #meaningful words the transcript have
        self.wordsetSize = 0  #all different words
        #init for updateKeywordStatistics()
        self.keywordStat = defaultdict(dict)  #num(int) certain keyword appears in the transcript
        self.keywordLoc = defaultdict(dict)  #locations(list) certain keyword appears in the transcript
        self.keywordPart = defaultdict(dict)  #percentage per keyword from all keywords
        self.keywordPer = float(0)  #percentage all keywords from all words
        #init for questionMark()
        self.sentNum = 0 
        self.questionStat = 0
        self.questionLoc = []
        self.questionPer = float(0)

        print("Reading data...")
        with open(fname, encoding='utf-8', errors='ignore') as f:
            data = f.read()
        tokenizer = nltk.tokenize.TreebankWordTokenizer()
        self.data = tokenizer.tokenize(data)
        print("done")

        print("Initializing keword dictionary...")
        # Dictionary of key-terms for CTS fidelity
        keywordCTS = defaultdict(set)

        keywordCTS['Agenda'] = {'agenda',['priorities','most important','focus on first'],['talk about today','work on today','focus on during the session','work on'] \
                                ,['you like to add to the agenda','you like to add anything to the agenda','you want to add to the agenda' \
                                ,'you want to add anything to the agenda'],'last week'}

        keywordCTS['Feedback'] = {'feedback',['previous','last time','last week','last session','past session'] \
                                  ,['think about today','things go today','think about today\'s session'],'concern','unhelpful','helpful' \
                                  ,['anything i can do better','anything we can do better'],['concerns about today\'s session','helpful about the session'] \
                                  ,'learn','skills','achieve','goals',['if i understand you correctly', 'are you saying', 'do i have it right']}

        keywordCTS['Understanding'] = {['understand','understanding'],'sounds like',['you are saying','you are feeling'],['you were feeling','you felt'] \
                                       ,['see','makes sense','i see'],['feel that way','feel this way']}

        keywordCTS['Interpersonal Effectiveness'] = {'sorry','hard','difficult','tough' \
                                                     ,'dissappointing','stressful','stressed' \
                                                     ,'scary','frightening','upset','upsetting'\
                                                     ,'unfortunate'}

        keywordCTS['Collaboration'] = {'choice', 'you want to do','good idea','because','will','help you get your goal'}

        keywordCTS['Guided Discovery'] = {'meaning','mean','self','how','why','evidence' \
                                          ,'conclusion','conclude','decide','decision','decided' \
                                          ,'know','proof','tell me more','assume','assumption' \
                                          ,'hypothesis','disprove','facts','fact','solutions' \
                                          ,'brainstorm','solve','alternative','other explanations' \
                                          ,'another way','other way','to think about','to explain','reason'} 

        keywordCTS['Focus on Key Cognitions'] = {'thinking','tell yourself','through your mind' \
                                                 ,'thought','think','connection','lead to','connected' \
                                                 ,'connect','link','linked','make you','you do'}

        keywordCTS['Choices of Intervention'] = {}

        keywordCTS['Homework'] = {'homework','review',' at home','practice','assignment','assign' \
                                  ,'assigned','progress','learned','improve','learn','skills' \
                                  ,'goal','better','barrier','in the way','expect','problems','succeed','success'}

        keywordCTS['Social Skills Training'] = {'rational','help you learn this skill','help you with your goal' \
                                                ,'demonstrate','to make your next role','play better','play even better','try to focus on'}
                                  

    def updateWordCount(self):
        print('Starting counting words...')
        stopWords = set(stopwords.words("english"))
        # Ignore capitalization and remove punctuation
        punctuation = set(string.punctuation)
        stemmer = PorterStemmer()
        for d in self.data:
            r = ''.join([c for c in d if not c in punctuation])
            if r != '':
                w = stemmer.stem(r.lower())
                self.words.append(w)
                self.transSize += 1
                if not w in stopWords:
                    self.wordCount[w] += 1
                    self.transSizeNS += 1
        self.wordsetSize = len(self.wordCount)
        self.wordCountSort = sorted(self.wordCount.items(), key = lambda kv: kv[1])
        print('Word counting done.')

    def updateKeywordStatistics(self):
        print('Starting keyword statisics...')
        s = 0
        for k1 in self.keywordCTS.keys():
            for k2 in self.keywordCTS[k1]:
                if k2 in self.wordCount.keys():
                    self.keywordStat[k1][k2] = self.wordCount[k2]
                    s += self.wordCount[k2]
                    loc = []
                    cap = len(self.words)
                    for i in range(cap):
                        if self.words[i] == k2:
                            loc.append(float(i)/float(cap))
                    self.keywordLoc[k1][k2] = loc
        for k1 in self.keywordStat.keys():
            for k2 in self.keywordStat[k1].keys():
                self.keywordPart[k1][k2] = self.keywordStat[k1][k2]/float(s)
        self.keywordPer = s/float(self.transSize)
        print('Keyword statistics done.')

    def questionMark(self):
    	print('Starting questionmark statistics...')
        with open(self.name, encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.lower()
                self.sentNum += len(sent_tokenize(line))
        l = len(self.data)
        for i in range(l):
    	    if self.data[i] == '?':
    	    	self.questionStat += 1
    	    	self.questionLoc.append(float(i)/float(l))
    	self.questionPer = float(self.questionStat)/float(self.sentNum)
    	print('Questionmark statistics done.')



class LabelledKeywordCount(KeywordCount):
    def __init__(self, fname, label):
        self.name = fname  #file name of input transcript
        self.label = label
        #init for updateWordCount()
        self.words = []  #sequenced word list
        self.wordCount = defaultdict(int)  #dictionary of words and their count
        self.wordCountSort = defaultdict(int)  #sorted dictionary
        self.transSize = 0  #words the transcript have
        self.transSizeNS = 0  #meaningful words the transcript have
        self.wordsetSize = 0  #all different words
        #init for updateKeywordStatistics()
        self.keywordStat = defaultdict(dict)  #num(int) certain keyword appears in the transcript
        self.keywordLoc = defaultdict(dict)  #locations(list) certain keyword appears in the transcript
        self.keywordPart = defaultdict(dict)  #percentage per keyword from all keywords
        self.keywordPer = float(0)  #percentage all keywords from all words

        print("Reading data...")
        from nltk.tokenize import sent_tokenize
        self.data = []
        self.lines = []
        with open(fname, encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.lower()
                if (line[0] == label):
                    self.lines.append(sent_tokenize(line))
                    tokenizer = nltk.tokenize.TreebankWordTokenizer()
                    self.data += tokenizer.tokenize(line)
        print("done")

        print("Initializing keyword dictionary...")
        # Dictionary of key-terms for CTS fidelity
        keywordCTS = defaultdict(set)

        keywordCTS['Agenda'] = {'agenda',['priorities','most important','focus on first'],['talk about today','work on today','focus on during the session','work on'] \
                                ,['you like to add to the agenda','you like to add anything to the agenda','you want to add to the agenda' \
                                ,'you want to add anything to the agenda'],'last week'}

        keywordCTS['Feedback'] = {'feedback',['previous','last time','last week','last session','past session'] \
                                  ,['think about today','things go today','think about today\'s session'],'concern','unhelpful','helpful' \
                                  ,['anything i can do better','anything we can do better'],['concerns about today\'s session','helpful about the session'] \
                                  ,'learn','skills','achieve','goals',['if i understand you correctly', 'are you saying', 'do i have it right']}

        keywordCTS['Understanding'] = {['understand','understanding'],'sounds like',['you are saying','you are feeling'],['you were feeling','you felt'] \
                                       ,['see','makes sense','i see'],['feel that way','feel this way']}

        keywordCTS['Interpersonal Effectiveness'] = {'sorry','hard','difficult','tough' \
                                                     ,'dissappointing','stressful','stressed' \
                                                     ,'scary','frightening','upset','upsetting'\
                                                     ,'unfortunate'}

        keywordCTS['Collaboration'] = {'choice', 'you want to do','good idea','because','will','help you get your goal'}

        keywordCTS['Guided Discovery'] = {'meaning','mean','self','how','why','evidence' \
                                          ,'conclusion','conclude','decide','decision','decided' \
                                          ,'know','proof','tell me more','assume','assumption' \
                                          ,'hypothesis','disprove','facts','fact','solutions' \
                                          ,'brainstorm','solve','alternative','other explanations' \
                                          ,'another way','other way','to think about','to explain','reason'} 

        keywordCTS['Focus on Key Cognitions'] = {'thinking','tell yourself','through your mind' \
                                                 ,'thought','think','connection','lead to','connected' \
                                                 ,'connect','link','linked','make you','you do'}

        keywordCTS['Choices of Intervention'] = {}

        keywordCTS['Homework'] = {'homework','review',' at home','practice','assignment','assign' \
                                  ,'assigned','progress','learned','improve','learn','skills' \
                                  ,'goal','better','barrier','in the way','expect','problems','succeed','success'}

        keywordCTS['Social Skills Training'] = {'rational','help you learn this skill','help you with your goal' \
                                                ,'demonstrate','to make your next role','play better','play even better','try to focus on'}
                                  
