import math, collections

class SmoothBigramModel:

  def __init__(self, corpus):
    """Initialize your data structures in the constructor."""
    self.smoothBigramCounts = collections.defaultdict(lambda: 0)
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
      for i in xrange(1, len(sentence.data)):
        datum = sentence.data[i]
        token = datum.word

        prevDatum = sentence.data[i-1]
        prevToken = prevDatum.word
        
        key = "%s, %s" % (prevToken, token) # first entry is current word; second entry is previous word
        self.smoothBigramCounts[key] += 1
        self.total += 1

  def score(self, sentence):
    """ Takes a list of strings as argument and returns the log-probability of the 
        sentence using your language model. Use whatever data you computed in train() here.
    """
    # COMPLETED your code here

    score = 0.0

    for i in xrange(1, len(sentence)):
      key = "%s, %s" % (sentence[i-1], sentence[i])
      count = self.smoothBigramCounts[key] + 1
      score += math.log(count)
      score -= math.log(self.total)
    
    return score
