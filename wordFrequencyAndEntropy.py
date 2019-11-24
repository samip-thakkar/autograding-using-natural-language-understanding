import scipy
import nltk
import string
from nltk.corpus import stopwords


### to get word frequency from string
def wordFrequencyAndEntropy(text):
	### String pre procesing  ##
	text = " ".join([c for c in text if c not in string.punctuation])
	text = [word for word in text.split() if word not in (stopwords.words('english'))]
	########
	d = {}
	for t in text:
		if(t in d.keys()):
			d[t] += 1
		else:
			d[t] = 1
	d = sorted(d.items(), key=lambda kv: kv[1])
	entropy = scipy.stats.entropy([i[1] for i in d])
	
	return [d, entropy]
	
