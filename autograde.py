import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import math
import nltk
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from monkeylearn import MonkeyLearn
import scipy
import string
analyser = SentimentIntensityAnalyzer()
nltk_sentiment = SentimentIntensityAnalyzer()

def sentiment_analyzer_scores(message):
    score = analyser.polarity_scores(message)
    return score

def calGrade(entropy, const, g_mean, word_count):
    ans = entropy * (2 * (const ** 0.5) + (g_mean ** 0.5)) * math.log(word_count)
    return ans
    

def removePunctutaions(stri):
    noPunc = []
    i=0
    while(i<len(stri)):
        if(stri[i] in string.punctuation):
            noPunc.append(" ")
        else:
            noPunc.append(stri[i])
        i += 1
    return "".join(noPunc)
### to get word frequency from string
def wordFrequencyAndEntropy(text):
    ### String pre procesing  ##
    text = removePunctutaions(text)
    text = [word for word in text.split() if word not in (stopwords.words('english'))]
    ########
    d = {}
    for t in text:
        if(t in d.keys()):
            d[t] += 1
        else:
            d[t] = 1
    d = sorted(d.items(), key=lambda kv: kv[1])
    total = np.sum([i[1] for i in d])
    entropy = scipy.stats.entropy([i[1] for i in d])
    custom_entropy = 0
    for i in d:
        custom_entropy -= (i[1] * np.log(i[1] / total))
    #custom_entropy = -1 * custom_entropy
    return [d, entropy, custom_entropy]

def dfToString(df):
    string = []
    df = df.to_list()
    for rows in df:
        string.append(rows)
    return ". ".join(string)

def readData(filePath):
    with open(filePath, 'r') as f:
        s = f.readline()
        s = s.split(',')
        for i in range(len(s)):
            s[i] = s[i].strip().lower()
    return s

def main(file_name, column_name):
    df = pd.read_excel(file_name , names=column_name)
    print("Reading inputs")
    df['name'] = df['name'].apply(lambda x: " ".join(x.split()[2:]))
    df.sort_values(by='name', ascending=True, inplace=True)
    df = df.groupby(by='name')['text'].apply(dfToString).reset_index()
    print("Input sorted")
    ############# pre-processed data frame #############
    
    entropy = []
    custom_entropy = []
    wc = []
    grade = []
    gmean = []
    #sentiment = []
    #sen1,sen2 = [],[]
    print("Calculating sentiment score and entropy")
    neg, neu, pos,critic_score= [],[],[],[]
    critic = readData('C:/Users/thakk/OneDrive/Desktop/Slack/Crit/LinkedIn Crit/critic.txt')
    for index,rows in df.iterrows():
        #print(index, len(rows.text))
        t = df.loc[index]['text']
        sentiment_score = sentiment_analyzer_scores(t)
        neg.append(sentiment_score['neg'])
        neu.append(sentiment_score['neu'])
        pos.append(sentiment_score['pos'])
        gmean.append((sentiment_score['neg'] * sentiment_score['pos']) ** 0.5)
        WordCount,e,cust_e = wordFrequencyAndEntropy(t)
        critic_count = 0
        
        for i, j in WordCount:
            if i in critic:
                critic_count += j    
        
        sum = 0
        for x in WordCount:
            sum += x[1]
        wc.append(sum)
        entropy.append(e)
        custom_entropy.append(cust_e)
        critic_score.append(critic_count / wc[-1])
        result = calGrade(e, critic_score[-1], gmean[-1], sum)
        grade.append(result)
        
        
    df['Entropy'] = entropy
    df['WordCount'] = wc
    df['custom_entropy'] = custom_entropy
    df['Negative'] = neg
    df['Neutral'] = neu
    df['Positive'] = pos
    df['Constructivity'] = critic_score
    df['Gmean'] = gmean
    df['Grade'] = grade
    
    print("Check output file")
    df.to_excel('output5.xlsx')
    
    import smtplib 
    from email.mime.multipart import MIMEMultipart 
    from email.mime.text import MIMEText 
    from email.mime.base import MIMEBase 
    from email import encoders 
      
    fromaddr = "cee300asu@gmail.com"
    toaddr = ["thakkarsamip1310@gmail.com", "samipthakkar1310@gmail.com"]
     
    for i in toaddr:
        msg = MIMEMultipart()
        msg['From'] = fromaddr 
        msg['To'] = i
        msg['Subject'] = "Mail with attachment"
        body = "PFA"
        msg.attach(MIMEText(body, 'plain')) 
        filename = "C:/Users/thakk/OneDrive/Desktop/Slack/Crit/LinkedIn Crit/output5.xlsx"
        
        attachment = open(filename, "rb") 
        p = MIMEBase('application', 'octet-stream') 
        p.set_payload((attachment).read()) 
        encoders.encode_base64(p) 
        p.add_header('Content-Disposition', "attachment; filename= %s" % filename) 
        msg.attach(p) 
        s = smtplib.SMTP('smtp.gmail.com', 587) 
        s.starttls() 
        s.login(fromaddr, "thomasseager") 
        text = msg.as_string() 
        s.sendmail(fromaddr, i, text) 
        s.quit() 
    print("Check your mails!!!!")    
if __name__ == '__main__':
    # data file name
    file_name = "C:/Users/thakk/OneDrive/Desktop/Slack/Crit/LinkedIn Crit/LinkedInCrit_data.xlsx"
    column_name = ['name', 'text']
    main(file_name, column_name)

