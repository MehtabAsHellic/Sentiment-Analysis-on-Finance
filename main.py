import sys
import os
import pandas as pd
from src.data_preprocessing.preprocess import preprocess_data, load_data
from src.models.lexicon_model import LexiconModel
from src.models.ml_model import MLModel
from src.models.unsupervised_model import UnsupervisedModel
from src.models.vader_model import VADERModel
from src.sentiment_analyzer.hybrid_analyzer import HybridSentimentAnalyzer
from src.dashboard.app import create_app
from src.data_preprocessing.financial_phrasebank_loader import load_financial_phrasebank

# Global variables to store data and models
preprocessed_df = None
ml_model = None
unsupervised_model = None
vader_model = None
initialized = False  # Flag to check if initialization has been done

def main():
    global preprocessed_df, ml_model, unsupervised_model, vader_model, initialized

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Define file paths
    raw_data_path = os.path.join('data', 'financial_news_data.csv')
    preprocessed_data_path = os.path.join('data', 'preprocessed_financial_news_data.csv')
    ml_model_path = os.path.join('data', 'ml_model.joblib')
    analyzed_data_path = os.path.join('data', 'analyzed_financial_news_data.csv')
    financial_phrasebank_path = os.path.join('data', 'financial_phrasebank.txt')

    # Load and preprocess data only if not already done
    if not initialized:  # Check if initialization has been done
        print("Loading and preprocessing data...")
        df = load_data(raw_data_path)
        if df is None:
            print("Error: Unable to load the raw data. Please check the file and try again.")
            return

        preprocessed_df = preprocess_data(df)
        if preprocessed_df is None:
            print("Error: Unable to preprocess the data. Please check the preprocessing step and try again.")
            return

        preprocessed_df.to_csv(preprocessed_data_path, index=False)
        print(f"Preprocessed data saved to {preprocessed_data_path}")

        #Financial PhraseBank dataset
        financial_phrasebank_df = load_financial_phrasebank(financial_phrasebank_path)

        # Train ML model
        print("Training ML model...")
        ml_model = MLModel()
        ml_model.train(financial_phrasebank_df['text'], financial_phrasebank_df['sentiment'])
        ml_model.save_model(ml_model_path)
        print(f"ML model saved to {ml_model_path}")

        # Train Unsupervised model
        print("Training Unsupervised model...")
        unsupervised_model = UnsupervisedModel()
        unsupervised_model.train(preprocessed_df['processed_text'])

        # Initialize VADER model
        print("Initializing VADER model...")
        vader_model = VADERModel()

        # Apply sentiment analysis to the dataframe
        print("Applying sentiment analysis...")
        preprocessed_df['rule_based_sentiment'] = preprocessed_df['processed_text'].apply(LexiconModel().analyze_sentiment)
        preprocessed_df['ml_sentiment'] = preprocessed_df['processed_text'].apply(ml_model.predict)
        preprocessed_df['unsupervised_sentiment'] = preprocessed_df['processed_text'].apply(unsupervised_model.predict)
        preprocessed_df['hybrid_sentiment'] = preprocessed_df['processed_text'].apply(HybridSentimentAnalyzer(ml_model_path).analyze_sentiment)
        preprocessed_df['sentiment_score'] = preprocessed_df['processed_text'].apply(HybridSentimentAnalyzer(ml_model_path).get_sentiment_score)
        preprocessed_df['vader_sentiment'] = preprocessed_df['processed_text'].apply(vader_model.analyze_sentiment)
        preprocessed_df['vader_score'] = preprocessed_df['processed_text'].apply(vader_model.get_sentiment_score)
        preprocessed_df.to_csv(analyzed_data_path, index=False)
        print(f"Analyzed data saved to {analyzed_data_path}")

        initialized = True 

    print("Starting the dashboard...")
    app = create_app()
    app.run_server(debug=False)

if __name__ == "__main__":
    main()