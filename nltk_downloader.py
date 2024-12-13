import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

def download_nltk_data():
    resources = [
        'punkt',
        'stopwords',
        'wordnet',
        'vader_lexicon',
        'averaged_perceptron_tagger',
        'punkt_tab'
    ]

    for resource in resources:
        print(f"Downloading {resource}...")
        nltk.download(resource)

if __name__ == "__main__":
    download_nltk_data()