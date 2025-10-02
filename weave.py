import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon', quiet=True)
sia = SentimentIntensityAnalyzer()
data = pd.DataFrame({'feedback': ['Love community events!', 'Need more fitness options.', 'Bored, suggest games.']})
data['sentiment'] = data['feedback'].apply(lambda x: sia.polarity_scores(x)['compound'])
def suggest_event(text):
    if 'fitness' in text.lower(): return 'Yoga Night'
    if 'game' in text.lower(): return 'Game Night'
    return 'Coffee Social'
data['suggestion'] = data['feedback'].apply(suggest_event)
print(data)