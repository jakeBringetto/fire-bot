import pdfplumber
import csv

# All data processing fns

def split_chunks(text, chunk_size):
    words = text.split()
    text_chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = ' '.join(words[i:i+chunk_size])
        text_chunks.append([chunk])
    return text_chunks   

# Aggregate a text file into sentence list
def file_to_csv(path):
    text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    chunks = split_chunks(text, 1000)
    with open('../data/data.csv', 'w') as csvfile:  
        csvwriter = csv.writer(csvfile)  
        csvwriter.writerows(chunks) 


# Called in user_query to create main data for query
def process_files():
    path = '../data/ERG2020.pdf'
    file_to_csv(path)

def csv_to_array():
    chunk_arr = []
    with open('../data/data.csv', 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        for line in csvreader:
            chunk_arr.append(line)
    return chunk_arr


if __name__ == "__main__":
    process_files()
    