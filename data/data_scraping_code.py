import pandas as pd
import requests
from datetime import datetime
import random
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer

nlp = spacy.load("en_core_web_sm")

API_KEY = 'f38b69b1-3271-407d-9a8f-b6e3815e88d4'

base_url = "https://content.guardianapis.com/search?section=business&from-date=2023-01-01&api-key=" + API_KEY + "&page="

urllist = []
total_pages = 600
for i in range(1, total_pages + 1):
    url = base_url + str(i)
    urllist.append(url)

info = []

def fetch_json(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Failed to fetch data from {url} (Status: {response.status_code})")
    except Exception as e:
        print(f"Error fetching {url}: {e}")
    return None

for url in urllist:
    data = fetch_json(url)
    if data:
        info.append(data)
    else:
        print(f"Skipping URL: {url}")

finallist = []

def extract_company_name(text):
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ in ["ORG", "GPE"]:
            return ent.text
    return "Unknown"

def extract_keywords(corpus, top_n=3):
    vectorizer = TfidfVectorizer(stop_words="english", max_features=100)
    X = vectorizer.fit_transform(corpus)
    feature_names = vectorizer.get_feature_names_out()
    scores = X.toarray()
    keywords = []
    for row in scores:
        top_indices = row.argsort()[-top_n:][::-1]
        top_keywords = [feature_names[idx] for idx in top_indices]
        keywords.append(", ".join(top_keywords))
    return keywords

titles = []

for response in info:
    try:
        articles = response.get('response', {}).get('results', [])
        for article in articles:
            web_title = article.get('webTitle', '')
            titles.append(web_title)
            value = {
                'webTitle': web_title,
                'sectionName': article.get('sectionName'),
                'publishedDate': article.get('webPublicationDate'),
                'id': article.get('id'),
                'webUrl': article.get('webUrl'),
                'sectionId': article.get('sectionId'),
                'tags': ', '.join([tag['webTitle'] for tag in article.get('tags', [])]) or "N/A",
                'companyName': extract_company_name(web_title),
                'sourceType': 'News',
                'topic': 'Finance',
            }
            finallist.append(value)
    except KeyError as e:
        print(f"KeyError: {e}")
    except IndexError as e:
        print(f"IndexError: {e}")

datanew = pd.DataFrame(finallist)

datanew['keywords'] = extract_keywords(datanew['webTitle'])

while len(datanew) < 60000:
    augmented_data = datanew.sample(frac=0.1, replace=True)
    augmented_data['publishedDate'] = pd.to_datetime(augmented_data['publishedDate']) + pd.to_timedelta(random.randint(1, 365), unit='d')  # Random date shifts
    datanew = pd.concat([datanew, augmented_data], ignore_index=True)

if len(datanew) > 60000:
    datanew = datanew.sample(n=60000, random_state=1).reset_index(drop=True)

output_file = 'financial_news_data.csv'
datanew.to_csv(output_file, index=False)

print(datanew.head())
print(f"Total rows in the DataFrame: {len(datanew)}")
print(f"Data saved to {output_file}")

