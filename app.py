from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import logging
from functools import wraps
import time
import os
import json
from difflib import SequenceMatcher
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Download necessary NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except Exception as e:
    pass  # Handle silently if running in an environment where downloads aren't possible

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MAX_TEXT_LENGTH = 10000  # Maximum allowed text length
RATE_LIMIT_REQUESTS = 100  # Number of requests allowed
RATE_LIMIT_WINDOW = 3600  # Time window in seconds (1 hour)
SIMILARITY_THRESHOLD = 90  # Threshold for success

# Rate limiting storage
request_history = {}

try:
    # Load the model globally so it's only loaded once
    # For Hindi and English, we use a multilingual model
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    raise

def rate_limit(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Simple IP-based rate limiting
        ip = request.remote_addr
        current_time = time.time()
        
        if ip in request_history:
            requests = [t for t in request_history[ip] 
                       if current_time - t < RATE_LIMIT_WINDOW]
            request_history[ip] = requests
            
            if len(requests) >= RATE_LIMIT_REQUESTS:
                return jsonify({'error': 'Rate limit exceeded'}), 429
                
        request_history.setdefault(ip, []).append(current_time)
        return f(*args, **kwargs)
    return decorated_function

def calculate_similarity(text1, text2):
    """
    Calculate semantic similarity between two texts using sentence transformers
    """
    # Generate embeddings
    embedding1 = model.encode([text1])
    embedding2 = model.encode([text2])
    
    # Calculate cosine similarity
    similarity = cosine_similarity(embedding1, embedding2)[0][0]
    
    # Convert to percentage
    similarity_percentage = float(np.round(similarity * 100, 2))
    
    return similarity_percentage

def identify_missing_content(user_text, reference_text, language="en"):
    """
    Identify what might be missing in the user's text compared to the reference text
    """
    # Determine language and set appropriate stopwords
    stop_words = set()
    try:
        if language.startswith("hi"):
            # Hindi stopwords if available
            if 'hindi' in stopwords.fileids():
                stop_words = set(stopwords.words('hindi'))
        else:
            # Default to English
            stop_words = set(stopwords.words('english'))
    except:
        # If there's an issue with stopwords, proceed without them
        pass
    
    # Tokenize both texts
    user_tokens = word_tokenize(user_text.lower())
    ref_tokens = word_tokenize(reference_text.lower())
    
    # Remove stopwords
    user_tokens = [word for word in user_tokens if word not in stop_words]
    ref_tokens = [word for word in ref_tokens if word not in stop_words]
    
    # Find missing words (present in reference but not in user text)
    missing_words = [word for word in ref_tokens if word not in user_tokens]
    
    # Group consecutive missing words to form phrases
    missing_phrases = []
    phrase = []
    
    for i, word in enumerate(ref_tokens):
        if word in missing_words:
            phrase.append(word)
        elif phrase:
            if len(phrase) > 0:
                missing_phrases.append(" ".join(phrase))
            phrase = []
    
    # Add the last phrase if it exists
    if phrase:
        missing_phrases.append(" ".join(phrase))
    
    # Return unique phrases
    return list(set(missing_phrases))

@app.route('/compare', methods=['POST'])
@rate_limit
def compare_texts():
    try:
        data = request.get_json()
        
        # Validate input
        if not data or 'text' not in data:
            return jsonify({
                'error': 'Missing required field. Please provide user text.'
            }), 400
            
        user_text = data['text']
        reference_texts = data.get('reference_texts', [])  # Get reference texts from request
        
        # Validate text content
        if not isinstance(user_text, str):
            return jsonify({
                'error': 'User text must be a string.'
            }), 400
            
        if not user_text.strip():
            return jsonify({
                'error': 'Empty text is not allowed.'
            }), 400
            
        # Check text length
        if len(user_text) > MAX_TEXT_LENGTH:
            return jsonify({
                'error': f'Text length exceeds maximum limit of {MAX_TEXT_LENGTH} characters.'
            }), 400
        
        # Validate reference texts
        if not isinstance(reference_texts, list):
            return jsonify({
                'error': 'Reference texts must be provided as an array.'
            }), 400
        
        if not reference_texts:
            return jsonify({
                'error': 'At least one reference text must be provided.'
            }), 400
            
        # Check each reference text
        for i, ref_text in enumerate(reference_texts):
            if not isinstance(ref_text, str):
                return jsonify({
                    'error': f'Reference text at index {i} must be a string.'
                }), 400
                
            if len(ref_text) > MAX_TEXT_LENGTH:
                return jsonify({
                    'error': f'Reference text at index {i} exceeds maximum length of {MAX_TEXT_LENGTH} characters.'
                }), 400
        
        # Compare user text with all reference texts
        best_match = {
            'similarity': 0,
            'reference_text': '',
            'missing_content': []
        }
        
        for ref_text in reference_texts:
            similarity = calculate_similarity(user_text, ref_text)
            
            if similarity > best_match['similarity']:
                # Determine language based on the reference text
                lang = "hi" if any(ord(c) >= 0x0900 and ord(c) <= 0x097F for c in ref_text) else "en"
                
                missing_content = []
                if similarity < SIMILARITY_THRESHOLD:
                    missing_content = identify_missing_content(user_text, ref_text, lang)
                
                best_match = {
                    'similarity': similarity,
                    'reference_text': ref_text,
                    'missing_content': missing_content
                }
        
        # Prepare response
        response = {
            'success': best_match['similarity'] >= SIMILARITY_THRESHOLD,
            'similarity_percentage': best_match['similarity'],
            'text_length': len(user_text),
            'reference_text': best_match['reference_text']
        }
        
        # Add missing content if similarity is below threshold
        if not response['success']:
            response['missing_content'] = best_match['missing_content']
            
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return jsonify({
            'error': 'An internal server error occurred.'
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """
    Simple health check endpoint
    """
    return jsonify({
        'status': 'healthy',
        'model': 'paraphrase-multilingual-MiniLM-L12-v2'
    })

if __name__ == '__main__':
    # Get port from environment variable or default to 8080
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)