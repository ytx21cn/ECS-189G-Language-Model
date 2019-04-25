import math, collections
class CustomModel:

  def __init__(self, corpus):
    """Initial custom language model and structures needed by this mode"""
    self.bigramCounts = collections.defaultdict(lambda: 0)
    self.unigramCounts = collections.defaultdict(lambda: 0)
    
    self.beforeWordCounts = collections.defaultdict(lambda: 0)
    self.beforeWordSums = collections.defaultdict(lambda: 0)
    self.afterWordCounts = collections.defaultdict(lambda: 0)
    self.afterWordSums = collections.defaultdict(lambda: 0)

    self.total = 0
    self.train(corpus)

  def train(self, corpus):
    """ Takes a corpus and trains your language model.
    """  
    # COMPLETED your code here

    # count unigrams and bigrams
    for sentence in corpus.corpus:
      data = sentence.data
      # process first for unigram, then for bigram
      for i in xrange(0, len(data)):
        self.total += 1
        token = data[i].word
        self.unigramCounts[token] += 1
        
        if i >= 1:
          prevToken = data[i-1].word
          key = "%s,%s" % (prevToken, token)
          self.bigramCounts[key] += 1

          if self.bigramCounts[key] == 1: # if new key found
            # update counts of unique before / after words
            self.afterWordCounts[prevToken] += 1
            self.beforeWordCounts[token] += 1

          # update the sums (occurences) for before / after words
          self.afterWordSums[prevToken] += 1
          self.beforeWordSums[token] += 1


  def score(self, sentence):
    """ With list of strings, return the log-probability of the sentence with language model. Use
        information generated from train.
    """
    # COMPLETED your code here

    score = 0

    for i in xrange(1, len(sentence)):
      d = 0.75
      
      currentWord = sentence[i]
      prevWord = sentence[i-1]
      bigramKey = "%s,%s" % (prevWord, currentWord)
      bigramCount = self.bigramCounts[bigramKey]

      if bigramCount > 0: # if bigram count works
        beforeWordCount = self.beforeWordCounts[currentWord]
        afterWordCount = self.afterWordCounts[prevWord]
        unigramCount = self.unigramCounts[prevWord]
        afterWordSum = self.afterWordSums[prevWord]

        if bigramCount == 1:
          d = 0.5

        # compute lambda(w_(i-1))
        lambdaPrev = (d / float(afterWordSum)) * float(afterWordCount)

        # compute P_continuation
        pContCurrent = float(beforeWordCount) / float(afterWordSum)

        currentScore = float(bigramCount - d) / float(afterWordSum) + lambdaPrev * (unigramCount / self.total)
        currentScore = math.log(currentScore)
        score += currentScore

      else: # if bigram does not work, try unigram instead
        alpha = 0.4
        unigramCount = self.unigramCounts[currentWord]
        if unigramCount > 0:
            score += math.log(unigramCount)
        else:
            score += math.log(0.00001)
        score -= math.log(self.total)
        score += math.log(alpha)

    return score
