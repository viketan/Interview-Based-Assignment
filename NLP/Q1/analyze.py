import csv
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from string import punctuation
import warnings
warnings.filterwarnings('ignore')

# Function to preprocess the comments
def preprocess_comments(comments):
    # Tokenize comments into individual words
    tokenized_comments = [nltk.word_tokenize(comment.lower()) for comment in comments]

    # Remove stopwords and punctuation
    stopwords = set(nltk.corpus.stopwords.words("english"))
    punctuations = set(punctuation)
    processed_comments = []
    for comment in tokenized_comments:
        processed_comment = [word for word in comment if word not in stopwords and word not in punctuation]
        processed_comments.append(processed_comment)

    return processed_comments

# Function to analyze comments using LDA
def analyze_comments(comments):
    # Preprocess comments
    processed_comments = preprocess_comments(comments)

    # Convert comments into document-term matrix
    vectorizer = CountVectorizer(tokenizer=lambda doc: doc, lowercase=False)
    dtm = vectorizer.fit_transform(processed_comments)

    # Apply LDA
    num_topics = 5  # Adjust the number of topics as per your preference
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda.fit(dtm)

    # Extract and print the most demanding topic
    feature_names = vectorizer.get_feature_names()
    topic_keywords = []
    for topic_idx, topic in enumerate(lda.components_):
        top_keywords_idx = topic.argsort()[:-6:-1]
        top_keywords = [feature_names[i] for i in top_keywords_idx]
        topic_keywords.append(top_keywords)

    return topic_keywords

# Main function to run the code
def main():
    # Read comments from the CSV file
    comments = []
    with open("youtube_comments.csv", "r", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip the header row
        for row in reader:
            comments.append(row[0])

    # Analyze comments using LDA
    topic_keywords = analyze_comments(comments)

    # Print the most demanding topic
    most_demanding_topic = topic_keywords[0]
    print("The most demanding topic is:", most_demanding_topic)

# Execute the main function
if __name__ == "__main__":
    main()
