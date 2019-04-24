import math, collections
class CustomModel:

  def __init__(self, corpus):
    """Initial custom language model and structures needed by this mode"""
    self.bigramCounts = collections.defaultdict(lambda: 0)
    self.unigramCounts = collections.defaultdict(lambda: 0)
    self.beforeWordCounts = collections.defaultdict(lambda: 0)
    self.afterWordCounts = collections.defaultdict(lambda: 0)
    self.afterWordLists = collections.defaultdict(lambda: [])
    self.total = 0
    self.train(corpus)

  def train(self, corpus):
    """ Takes a corpus and trains your language model.
    """  
    # TODO your code here
    
    for sentence in corpus.corpus:
      data = sentence.data
      # process first for unigram, then for bigram
      for i in xrange(0, len(data)):
        self.total += 1
        token = data[i].word
        self.unigramCounts[token] += 1
        if i >= 1:
          prevToken = data[i-1].word
          key = "%s, %s" % (prevToken, token)
          self.bigramCounts[key] += 1
        if i < (len(data) - 1):
          nextToken = data[i+1].word
          key = "%s, %s" % (token, nextToken)
          self.afterWordLists[token].append(key)
    
    for token in self.unigramCounts.keys():
      self.afterWordCounts[token] = self.countAfterWords(self.bigramCounts, token)
      self.beforeWordCounts[token] = self.countBeforeWords(self.bigramCounts, token)

  def score(self, sentence):
    """ With list of strings, return the log-probability of the sentence with language model. Use
        information generated from train.
    """
    # TODO your code here

    score = 0
    d = 0.75

    for i in xrange(0, len(sentence)):
      if i >= 1:
        currentWord = sentence[i]
        prevWord = sentence[i-1]
        bigramKey = "%s, %s" % (prevWord, currentWord)
        bigramCount = self.bigramCounts[bigramKey]

        if bigramCount > 0: # if bigram count works
          beforeWordCount = self.beforeWordCounts[currentWord]
          afterWordCount = self.afterWordCounts[prevWord]
          unigramCount = self.unigramCounts[prevWord]

          # compute lambda(w_(i-1))
          lambdaPrev = 0
          if unigramCount == 0:
            lambdaPrev = d * 0.1
          else:
            afterWordSum = self.sumAfterWords(self.bigramCounts, prevWord)
            lambdaPrev = (d / float(afterWordSum)) * float(afterWordCount)

          # compute P_continuation
          pContCurrent = float(beforeWordCount) / float(len(self.bigramCounts))

          currentScore = float(max(bigramCount - d, 0)) / float(unigramCount) + lambdaPrev * pContCurrent
          currentScore = math.log(currentScore)
          score += currentScore

        else: # if bigram does not work, try unigram instead
          unigramCount = self.unigramCounts[currentWord]
          
          score += math.log(unigramCount + 1)
          score -= math.log(self.total)
          score += math.log(0.5)

    return score
          

  def countBeforeWords(self, dictionary, token):
    ''' count the number of unique words at position i-1 that come before word i
    '''
    num = 0
    keys = dictionary.keys()
    for key in keys:
      if key.endswith(token):
        num += 1
    return num

  def countAfterWords(self, dictionary, token):
    ''' count the number of unique words at position i that come after word i-1
    '''
    num = 0
    keys = dictionary.keys()
    for key in keys:
      if key.startswith(token):
        num += 1
    return num

  def sumAfterWords(self, countDict, token):
    ''' sum the counts of words at position i that come after word i-1
    '''
    sum = 0
    keys = self.afterWordLists[token]
    for key in keys:
      sum += countDict[key]
    return sum
