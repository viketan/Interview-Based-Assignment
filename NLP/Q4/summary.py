import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

def read_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

def preprocess_text(text):
    # Tokenize the text into sentences
    sentences = sent_tokenize(text)
    
    # Tokenize each sentence into words
    words = [word_tokenize(sentence) for sentence in sentences]
    
    # Remove stopwords and punctuation
    stop_words = set(stopwords.words('english'))
    words = [[word for word in sentence if word.lower() not in stop_words and word.isalnum()] for sentence in words]
    
    return sentences, words

def generate_summary(file_path, num_sentences=3):
    # Read the text file
    text = read_text(file_path)
    
    # Preprocess the text
    sentences, words = preprocess_text(text)
    
    # Create the TF-IDF matrix
    documents = [' '.join(sentence) for sentence in words]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)
    
    # Apply Latent Semantic Analysis (LSA)
    lsa_model = TruncatedSVD(n_components=num_sentences)
    lsa_matrix = lsa_model.fit_transform(tfidf_matrix)
    
    # Calculate sentence scores based on LSA weights
    sentence_scores = lsa_matrix.sum(axis=1)
    
    # Select the top sentences with highest scores as the summary
    ranked_sentences = [(sentence, score) for sentence, score in zip(sentences, sentence_scores)]
    ranked_sentences.sort(key=lambda x: x[1], reverse=True)
    summary_sentences = [sentence for sentence, _ in ranked_sentences[:num_sentences]]
    summary = ' '.join(summary_sentences)
    
    return summary

# Example usage
def main():
    file_path = 'article.txt'
    summary = generate_summary(file_path, num_sentences=3)
    print(summary)

if __name__=="__main__":
    main()