import math, collections

class BackoffModel:

  def __init__(self, corpus):
    """Initialize your data structures in the constructor."""
    self.bigramCounts = collections.defaultdict(lambda: 0)
    self.smoothUnigramCounts = collections.defaultdict(lambda: 0)
    self.total = 0
    self.train(corpus)

  def train(self, corpus):
    """ Takes a corpus and trains your language model. 
        Compute any counts or other corpus statistics in this function.
    """  
    # COMPLETED your code here
    # Tip: To get words from the corpus, try
    #    for sentence in corpus.corpus:
    #       for datum in sentence.data:  
    #         word = datum.word

    for sentence in corpus.corpus:
      for i in xrange(0, len(sentence.data)):
        # first add the training entry for smooth unigram model
        datum = sentence.data[i]
        token = datum.word
        self.smoothUnigramCounts[token] += 1
        self.total += 1
        
        # then add the training entry for (raw) bigram as well
        if i > 0:
          precDatum = sentence.data[i-1]
          precToken = precDatum.word
          self.bigramCounts[(token, precToken)] += 1

  def score(self, sentence):
    """ Takes a list of strings as argument and returns the log-probability of the 
        sentence using your language model. Use whatever data you computed in train() here.
    """
    # COMPLETED your code here

    score = 0.0
    
    for i in xrange(1, len(sentence)):
      currentWord = sentence[i]
      precWord = sentence[i-1]
      currentTuple = (currentWord, precWord)

      bigramCount = self.bigramCounts[currentTuple]
      smoothUnigramCount = self.smoothUnigramCounts[currentWord]
      
      if bigramCount > 0:
        score += math.log(bigramCount)
        score -= math.log(smoothUnigramCount)
      else: # try unigram model if count == 0
        alpha = 0.4
        score += math.log(smoothUnigramCount + 1)
        score -= math.log(self.total * 3)
        score += math.log(alpha)
        
    return score


    # then, if the score for raw bigram model is 0, score it again with the smooth unigram model
