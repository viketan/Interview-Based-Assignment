import csv
from collections import Counter
from PyPDF2 import PdfReader

def extract_text_from_pdf(pdf_path):
    # Open the PDF file
    with open(pdf_path, 'rb') as file:
        reader = PdfReader(file)
        num_pages = len(reader.pages)

        text = ''
        for page_number in range(num_pages):
            page = reader.pages[page_number]
            text += page.extract_text()

        # Remove newline characters and extra whitespaces
        text = text.replace('\n', ' ').strip()
        return text

def store_text_in_csv(text, csv_path):
    with open(csv_path, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Text'])
        writer.writerow([text])

def find_most_repeated_word(text):
    # Tokenize the text into individual words
    words = text.split()

    # Count the frequency of each word
    word_counts = Counter(words)

    # Find the most repeated word
    most_repeated_word = word_counts.most_common(1)[0][0]
    return most_repeated_word

# Main function to run the code
def main():
    pdf_path = 'Viketan_ Revankar.pdf'
    csv_path = 'pdf_text.csv'
    # Extract text from the PDF
    text = extract_text_from_pdf(pdf_path)

    if text:
        # Store the extracted text in a CSV file
        store_text_in_csv(text, csv_path)

        # Find the most repeated word
        most_repeated_word = find_most_repeated_word(text)

        print("Text extracted and saved in 'pdf_text.csv'.")
        print("The most repeated word is:", most_repeated_word)
    else:
        print("No text found in the PDF.")

# Execute the main function
if __name__ == '__main__':
    main()
