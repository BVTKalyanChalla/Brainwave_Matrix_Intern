import pandas as pd
import numpy as np
import praw
from newsapi import NewsApiClient
import nltk
import spacy
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import LdaModel
from gensim.corpora import Dictionary
from collections import Counter
import os
from dotenv import load_dotenv
import logging
import re

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataCollector:
    def __init__(self):
        self.reddit = self._init_reddit()
        self.newsapi = self._init_newsapi()

    def _init_reddit(self):
        try:
            return praw.Reddit(
                client_id=os.getenv('REDDIT_CLIENT_ID'),
                client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
                user_agent=os.getenv('REDDIT_USER_AGENT')
            )
        except Exception as e:
            logger.error(f"Error initializing Reddit API: {e}")
            return None

    def _init_newsapi(self):
        try:
            return NewsApiClient(api_key=os.getenv('NEWS_API_KEY'))
        except Exception as e:
            logger.error(f"Error initializing News API: {e}")
            return None

    def collect_reddit_data(self, subreddit_name, limit=100):
        if not self.reddit:
            return pd.DataFrame()

        posts_list = []
        try:
            subreddit = self.reddit.subreddit(subreddit_name)
            for post in subreddit.hot(limit=limit):
                posts_list.append({
                    'date': datetime.fromtimestamp(post.created_utc),
                    'title': post.title,
                    'text': f"{post.title} {post.selftext}",
                    'source': 'reddit',
                    'url': post.url,
                    'score': post.score
                })
        except Exception as e:
            logger.error(f"Error collecting Reddit data: {e}")
        
        return pd.DataFrame(posts_list)

    def collect_news_data(self, query, days_back=7):
        if not self.newsapi:
            return pd.DataFrame()

        articles_list = []
        try:
            from_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
            articles = self.newsapi.get_everything(
                q=query,
                from_param=from_date,
                language='en',
                sort_by='relevancy'
            )

            for article in articles['articles']:
                articles_list.append({
                    'date': article['publishedAt'],
                    'title': article['title'],
                    'text': f"{article['title']} {article['description'] or ''}",
                    'source': 'news',
                    'url': article['url']
                })
        except Exception as e:
            logger.error(f"Error collecting news data: {e}")

        return pd.DataFrame(articles_list)

class TextProcessor:
    def __init__(self):
        # Download required NLTK data
        nltk_packages = ['punkt', 'averaged_perceptron_tagger', 
                        'maxent_ne_chunker', 'words', 'stopwords', 
                        'wordnet', 'vader_lexicon']
        for package in nltk_packages:
            try:
                nltk.download(package, quiet=True)
            except Exception as e:
                logger.error(f"Error downloading NLTK package {package}: {e}")

        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.nlp = spacy.load('en_core_web_sm')

    def preprocess(self, text):
        if not isinstance(text, str):
            return ""

        try:
            # Lowercase conversion
            text = text.lower()
            
            # Remove URLs
            text = re.sub(r'https?://\S+|www\.\S+', '', text)
            
            # Remove special characters
            text = re.sub(r'[^\w\s]', '', text)
            
            # Tokenization
            tokens = word_tokenize(text)
            
            # Remove stopwords and lemmatize
            tokens = [self.lemmatizer.lemmatize(token) 
                     for token in tokens 
                     if token not in self.stop_words]
            
            return ' '.join(tokens)
        except Exception as e:
            logger.error(f"Error in text preprocessing: {e}")
            return text

    def extract_features(self, text):
        try:
            doc = self.nlp(text)
            return {
                'entities': [(ent.text, ent.label_) for ent in doc.ents],
                'noun_phrases': [chunk.text for chunk in doc.noun_chunks],
                'pos_tags': [(token.text, token.pos_) for token in doc]
            }
        except Exception as e:
            logger.error(f"Error in feature extraction: {e}")
            return {'entities': [], 'noun_phrases': [], 'pos_tags': []}

class SentimentAnalyzer:
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()

    def analyze(self, text):
        try:
            # VADER sentiment
            vader_scores = self.sia.polarity_scores(text)
            
            # TextBlob analysis
            blob = TextBlob(text)
            
            return {
                'compound_score': vader_scores['compound'],
                'sentiment': self._get_sentiment_label(vader_scores['compound']),
                'subjectivity': blob.sentiment.subjectivity,
                'vader_scores': vader_scores
            }
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
            return None

    def _get_sentiment_label(self, compound_score):
        if compound_score >= 0.05:
            return 'positive'
        elif compound_score <= -0.05:
            return 'negative'
        else:
            return 'neutral'

class TopicModeler:
    def __init__(self, num_topics=5):
        self.num_topics = num_topics
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            max_df=0.95,
            min_df=2
        )

    def extract_topics(self, texts):
        try:
            # Create document-term matrix
            dtm = self.vectorizer.fit_transform(texts)
            
            # Create LDA model
            lda_model = LdaModel(
                corpus=[[(idx, val) for idx, val in enumerate(doc)] 
                       for doc in dtm.toarray()],
                num_topics=self.num_topics,
                id2word={idx: term for idx, term 
                        in enumerate(self.vectorizer.get_feature_names_out())},
                random_state=42
            )
            
            return [dict(lda_model.show_topic(topic_idx)) 
                   for topic_idx in range(self.num_topics)]
        except Exception as e:
            logger.error(f"Error in topic modeling: {e}")
            return []

class Visualizer:
    @staticmethod
    def plot_sentiment_distribution(df):
        plt.figure(figsize=(10, 6))
        sns.countplot(data=df, x='sentiment')
        plt.title('Sentiment Distribution')
        plt.xlabel('Sentiment')
        plt.ylabel('Count')
        plt.show()

    @staticmethod
    def plot_sentiment_by_source(df):
        plt.figure(figsize=(10, 6))
        sns.countplot(data=df, x='sentiment', hue='source')
        plt.title('Sentiment Distribution by Source')
        plt.xlabel('Sentiment')
        plt.ylabel('Count')
        plt.legend(title='Source')
        plt.show()

    @staticmethod
    def plot_sentiment_over_time(df):
        df['date'] = pd.to_datetime(df['date'])
        sentiment_by_date = df.groupby([df['date'].dt.date, 'sentiment']).size().unstack()
        
        plt.figure(figsize=(12, 6))
        sentiment_by_date.plot(kind='line', marker='o')
        plt.title('Sentiment Trends Over Time')
        plt.xlabel('Date')
        plt.ylabel('Count')
        plt.legend(title='Sentiment')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

class ContentAnalysisPipeline:
    def __init__(self):
        self.data_collector = DataCollector()
        self.text_processor = TextProcessor()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.topic_modeler = TopicModeler()
        self.visualizer = Visualizer()

    def run_analysis(self, query, source_type='all', limit=100):
        # Collect data
        data = self._collect_data(query, source_type, limit)
        if data.empty:
            logger.error("No data collected")
            return None

        # Process and analyze
        results = self._analyze_data(data)
        
        # Visualize results
        self._visualize_results(results)
        
        return results

    def _collect_data(self, query, source_type, limit):
        all_data = []
        
        if source_type in ['all', 'reddit']:
            reddit_data = self.data_collector.collect_reddit_data(
                subreddit_name=query.replace(" ", ""),
                limit=limit
            )
            if not reddit_data.empty:
                all_data.append(reddit_data)

        if source_type in ['all', 'news']:
            news_data = self.data_collector.collect_news_data(query)
            if not news_data.empty:
                all_data.append(news_data)

        return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()

    def _analyze_data(self, df):
        results = []
        
        for _, row in df.iterrows():
            # Preprocess text
            processed_text = self.text_processor.preprocess(row['text'])
            
            # Extract features
            features = self.text_processor.extract_features(row['text'])
            
            # Analyze sentiment
            sentiment = self.sentiment_analyzer.analyze(processed_text)
            
            results.append({
                'date': row['date'],
                'original_text': row['text'],
                'processed_text': processed_text,
                'sentiment': sentiment['sentiment'],
                'compound_score': sentiment['compound_score'],
                'subjectivity': sentiment['subjectivity'],
                'entities': features['entities'],
                'source': row['source'],
                'url': row['url']
            })
        
        results_df = pd.DataFrame(results)
        
        # Add topic modeling results
        topics = self.topic_modeler.extract_topics(results_df['processed_text'])
        results_df['topics'] = [topics] * len(results_df)
        
        return results_df

    def _visualize_results(self, df):
        self.visualizer.plot_sentiment_distribution(df)
        self.visualizer.plot_sentiment_by_source(df)
        self.visualizer.plot_sentiment_over_time(df)

def main():
    print("Content Sentiment Analysis Tool")
    print("------------------------------")
    
    # Get user input
    query = input("Enter search topic: ")
    source_type = input("Enter source type (reddit/news/all): ").lower()
    try:
        limit = int(input("Enter number of items to analyze (10-100): "))
    except ValueError:
        limit = 50
        print("Using default limit of 50")

    # Initialize and run pipeline
    pipeline = ContentAnalysisPipeline()
    results = pipeline.run_analysis(query, source_type, limit)

    if results is not None:
        # Save results
        output_file = f'sentiment_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        results.to_csv(output_file, index=False)
        print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    main()
