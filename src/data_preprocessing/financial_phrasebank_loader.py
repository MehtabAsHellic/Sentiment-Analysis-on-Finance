import pandas as pd

def load_financial_phrasebank(file_path):
    with open(file_path, 'r', encoding='ISO-8859-1') as file:  # Changed encoding here
        lines = file.readlines()
    
    data = []
    for line in lines:
        sentence, sentiment = line.strip().split('@')
        data.append({'text': sentence, 'sentiment': sentiment})
    
    df = pd.DataFrame(data)
    df['sentiment'] = df['sentiment'].map({'positive': 'Positive', 'negative': 'Negative', 'neutral': 'Neutral'})
    return df

if __name__ == "__main__":
    # Test the loader
    df = load_financial_phrasebank('path/to/financial_phrasebank.txt')
    print(df.head())
    print(df['sentiment'].value_counts())
