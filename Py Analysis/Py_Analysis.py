import os
import nltk
import pyodbc
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download the VADER lexicon for sentiment analysis if not already present.
nltk.download('vader_lexicon')

# Initialize the VADER sentiment intensity analyzer
sia = SentimentIntensityAnalyzer()

# Function to fetch data from a SQL database
def fetch_data_from_sql():
    """
    Connects to the SQL database, fetches data from the 'customer_reviews' table, 
    and returns it as a Pandas DataFrame.
    """
    conn_str = (
        "Driver={SQL Server};"
        "Server=MOHAMEDGADIA\\SQLEXPRESS;"
        "Database=Marketing_Analysis;"
        "Trusted_Connection=yes;"
    )
    with pyodbc.connect(conn_str) as conn:
        query = "SELECT ReviewID, CustomerID, ProductID, ReviewDate, Rating, ReviewText FROM dbo.customer_reviews"
        df = pd.read_sql(query, conn)
    return df

# Function to calculate sentiment scores
def calculate_sentiment(review):
    """
    Analyzes the sentiment of the given review text using VADER and 
    returns the compound sentiment score.
    """
    return sia.polarity_scores(review)['compound']

# Function to categorize sentiment based on score and rating
def categorize_sentiment(score, rating):
    """
    Categorizes the sentiment based on the compound score and review rating.
    """
    if score > 0.05:
        return 'Positive' if rating >= 4 else 'Mixed Positive' if rating == 3 else 'Mixed Negative'
    elif score < -0.05:
        return 'Negative' if rating <= 2 else 'Mixed Negative' if rating == 3 else 'Mixed Positive'
    else:
        return 'Neutral' if rating == 3 else 'Positive' if rating >= 4 else 'Negative'

# Function to bucket sentiment scores into predefined ranges
def sentiment_bucket(score):
    """
    Buckets the compound sentiment score into predefined textual ranges.
    """
    if score >= 0.5:
        return '0.5 to 1.0'
    elif 0.0 <= score < 0.5:
        return '0.0 to 0.49'
    elif -0.5 <= score < 0.0:
        return '-0.49 to 0.0'
    else:
        return '-1.0 to -0.5'

# Main analysis function
def analyze_reviews():
    """
    Fetches customer reviews data, applies sentiment analysis, and saves
    the results to a CSV file.
    """
    # Fetch data
    customer_reviews_df = fetch_data_from_sql()

    # Apply sentiment analysis
    customer_reviews_df['SentimentScore'] = customer_reviews_df['ReviewText'].apply(calculate_sentiment)

    # Apply sentiment categorization
    customer_reviews_df['SentimentCategory'] = customer_reviews_df.apply(
        lambda row: categorize_sentiment(row['SentimentScore'], row['Rating']), axis=1)

    # Apply sentiment bucketing
    customer_reviews_df['SentimentBucket'] = customer_reviews_df['SentimentScore'].apply(sentiment_bucket)

    # Save the results to a CSV file
    output_path = os.path.join('E:', 'projects', 'Marketing Analysis', 'Py Analysis', 'dbo_customer_reviews_with_sentiment.csv')
    customer_reviews_df.to_csv(output_path, index=False)

    # Display a preview of the data
    print(customer_reviews_df.head())