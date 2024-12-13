from src.models.custom_lexicon import get_sentiment, get_sentiment_score

class LexiconModel:
    def __init__(self):
        pass

    def analyze_sentiment(self, text):
        return get_sentiment(text)

    def get_sentiment_score(self, text):
        return get_sentiment_score(text)

if __name__ == "__main__":
    lexicon_model = LexiconModel()
    
    test_texts = [
        "The company's profits soared, exceeding all expectations.",
        "The stock market crashed, wiping out billions in value.",
        "The financial report showed mixed results for the quarter."
    ]
    
    for text in test_texts:
        sentiment = lexicon_model.analyze_sentiment(text)
        score = lexicon_model.get_sentiment_score(text)
        print(f"Text: {text}")
        print(f"Sentiment: {sentiment}")
        print(f"Score: {score}\n")