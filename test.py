import nltk
from nltk.tokenize import word_tokenize

# Add this line if running in Flask or a virtual environment
nltk.data.path.append(r'C:\Users\Anand\AppData\Roaming\nltk_data')

text = "This is a test sentence."
tokens = word_tokenize(text)
print(tokens)
