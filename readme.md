# **Sentiment Analysis**

## **Overview**
This project implements a **hybrid sentiment analysis model** combining rule-based and machine learning approaches to analyze financial news data. The model is compared against the pre-trained **VADER sentiment analysis model**, with insights displayed in an interactive dashboard built using **Dash**.

The dataset consists of real-time financial news scraped from The Guardian API, cleaned, and preprocessed for sentiment analysis. The dashboard visualizes trends, comparisons, and key insights in a user-friendly manner.

---

## **Features**
1. **Data Collection**:
   - Scrapes real-time financial news using the Guardian API.
   - Processes 60,000 data entries.

2. **Sentiment Analysis Models**:
   - **Hybrid Model**: Combines a rule-based lexicon and a machine learning model.
   - **VADER Sentiment Analyzer**: Used for performance comparison.

3. **Dashboard**:
   - Interactive visualizations of sentiment trends, model comparisons, and company-specific insights.
   - Built using Dash for real-time updates.

4. **Preprocessing**:
   - Text cleaning, tokenization, and stopword removal.
   - Feature extraction using **TF-IDF**.

## **Setup Instructions**

### **1. Prerequisites**
- **Python 3.8 or higher** installed on your system.
- **pip** (Python package installer).

### **2. Installation Steps**
1. **Unzip the Project Folder**:
   - Extract the project files from the ZIP archive.
   - Navigate to the project directory in your terminal.

2. **Install Required Libraries**:
   - Run the following command to install all dependencies:
     ```bash
     pip install -r requirements.txt
     ```

3. **Verify Installation**:
   - Ensure all necessary libraries, such as Pandas, Dash, and Scikit-learn, are installed.

### **3. Running the Project**
1. **Preprocess the Data**:
   - Navigate to the `src/data_preprocessing` directory.
   - Execute `preprocess.py` to clean and preprocess the dataset:
     ```bash
     python preprocess.py
     ```
 **Preprocess the Data** *(Already Completed, above is just a reference of how we did it. We can directly launch the dashboard by following the command)*:
   - The dataset has already been cleaned and preprocessed. No action is required here.

2. **Train the Sentiment Models**:
   - Train the machine learning model and run the VADER analyzer:
     ```bash
     python src/models/ml_model.py
     python src/models/vader_model.py
     ```
**Train the Sentiment Models** *(Already Completed, above is just a reference of how we did it. We can directly launch the dashboard by following the command)*:
   - The machine learning model and VADER analyzer have already been trained and set up. No further action is needed for training.

3. **Launch the Dashboard**:
   - From the project directory, run the following command:
     ```bash
     python run_app.py
     ```
   - Open your web browser and go to `http://127.0.0.1:8050/` to access the dashboard.

---

## **Key Components**
1. **Data Collection**:
   - Guardian API provides real-time financial news.
   - Data saved in `data/financial_news_data.csv`.

2. **Sentiment Analysis Models**:
   - **Rule-based Lexicon Model**: Uses a financial lexicon for sentiment scoring.
   - **Machine Learning Model**: Trained on the Financial PhraseBank dataset.
   - **VADER**: A pre-trained sentiment analyzer.

3. **Dashboard**:
   - Interactive charts and sentiment analysis comparisons.
   - Built using **Dash** for real-time updates.

---

## **Technical Details**
### Libraries Used:
- **Data Processing**: Pandas, NumPy
- **Text Preprocessing**: NLTK, Scikit-learn
- **Sentiment Analysis**:
  - VADER Sentiment
  - TF-IDF for feature extraction
  - Multinomial Naive Bayes
- **Dashboard Development**: Dash, Plotly

This README file provides a complete guide for setting up, running, and understanding the project. Let me know if any further refinements are required!