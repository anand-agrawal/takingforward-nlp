from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import logging
from functools import wraps
import time
import os
from textblob import TextBlob


app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MAX_TEXT_LENGTH = 10000  # Maximum allowed text length
RATE_LIMIT_REQUESTS = 100  # Number of requests allowed
RATE_LIMIT_WINDOW = 3600  # Time window in seconds (1 hour)

# Rate limiting storage
request_history = {}

try:
    # Load the model globally so it's only loaded once
    model = SentenceTransformer('all-MiniLM-L6-v2')
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

def correct_spelling(text):
    try:
        return str(TextBlob(text).correct())
    except Exception:
        return text  # fallback if correction fails

def preprocess_text(text):
    # Lowercase, strip, and correct spelling
    text = text.lower().strip()
    text = correct_spelling(text)
    return text

def calculate_similarity(text1, text2):
    """
    Calculate semantic similarity between two texts using sentence transformers
    """
    # Preprocess both texts
    # text1 = preprocess_text(text1)
    text2 = preprocess_text(text2)
    print(f"Comparing:\nText2: {text2}")

    # Generate embeddings
    embedding1 = model.encode([text1])
    embedding2 = model.encode([text2])
    
    # Calculate cosine similarity
    similarity = cosine_similarity(embedding1, embedding2)[0][0]
    
    # Convert to percentage
    similarity_percentage = float(np.round(similarity * 100, 2))
    
    return similarity_percentage


@app.route('/compare', methods=['POST'])
@rate_limit
def compare_texts():
    try:
        data = request.get_json()
        
        # Validate input
        if not data or 'text1' not in data or 'text2' not in data:
            return jsonify({
                'error': 'Missing required fields. Please provide text1 and text2.'
            }), 400
            
        text1 = data['text1']
        text2 = data['text2']
        
        # Validate text content
        if not isinstance(text1, str) or not isinstance(text2, str):
            return jsonify({
                'error': 'Both text1 and text2 must be strings.'
            }), 400
            
        if not text1.strip() or not text2.strip():
            return jsonify({
                'error': 'Empty text is not allowed.'
            }), 400
            
        # Check text length
        if len(text1) > MAX_TEXT_LENGTH or len(text2) > MAX_TEXT_LENGTH:
            return jsonify({
                'error': f'Text length exceeds maximum limit of {MAX_TEXT_LENGTH} characters.'
            }), 400
        
        # Calculate similarity
        similarity = calculate_similarity(text1, text2)
        
        return jsonify({
            'similarity_percentage': similarity,
            'text1_length': len(text1),
            'text2_length': len(text2)
        })
        
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return jsonify({
            'error': 'An internal server error occurred.'
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)