from flask import Flask, render_template, request, jsonify
import numpy as np 
import joblib 
import os
import sys
import time
from openai import OpenAI
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Global variable to store embeddings
df = None

def load_embeddings():
    """Load embeddings from file."""
    global df
    if not os.path.exists('embeddings.joblib'):
        raise FileNotFoundError("embeddings.joblib file not found. Please run preprocess.py first.")
    
    print("Loading embeddings...")
    df = joblib.load('embeddings.joblib')
    print(f"Loaded {len(df)} embeddings successfully")
    return df

def create_embedding(text_list, max_retries=3):
    """Create embeddings using OpenAI API with error handling and retry logic."""
    for attempt in range(max_retries):
        try:
            print(f"Creating embedding (attempt {attempt + 1}/{max_retries})...")
            
            # Use OpenAI embeddings API
            response = client.embeddings.create(
                model=os.getenv('OPENAI_EMBEDDING_MODEL', 'text-embedding-3-small'),
                input=text_list,
                dimensions=1024
            )
            
            # Extract embeddings from response
            embeddings = [data.embedding for data in response.data]
            print("Embedding created successfully")
            return embeddings
            
        except Exception as e:
            print(f"Error creating embeddings (attempt {attempt + 1}): {e}")
            if attempt == max_retries - 1:
                raise Exception("All retry attempts failed. Please check your OpenAI API key and try again.")
            time.sleep(2)  # Wait before retry

def inference(prompt, max_retries=3):
    """Generate response using OpenAI API with error handling and retry logic."""
    for attempt in range(max_retries):
        try:
            print(f"Generating response (attempt {attempt + 1}/{max_retries})...")
            
            # Use OpenAI chat completions API
            response = client.chat.completions.create(
                model=os.getenv('OPENAI_CHAT_MODEL', 'gpt-3.5-turbo'),
                messages=[
                    {"role": "system", "content": "You are a helpful mathematics tutor. Answer questions about geometry and transformations based on the provided video content."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=int(os.getenv('OPENAI_MAX_TOKENS', 1000)),
                temperature=float(os.getenv('OPENAI_TEMPERATURE', 0.7))
            )
            
            # Extract response content
            response_text = response.choices[0].message.content
            print("Generated response successfully")
            return {"response": response_text}
            
        except Exception as e:
            print(f"Error generating response (attempt {attempt + 1}): {e}")
            if attempt == max_retries - 1:
                raise Exception("All retry attempts failed. Please check your OpenAI API key and try again.")
            time.sleep(2)  # Wait before retry

def create_fallback_response(embeddings_data, query):
    """Create a simple fallback response when the generation API is unavailable."""
    response_parts = []
    response_parts.append(f"Based on your question '{query}', I found the following relevant video content:\n")
    
    for i, row in enumerate(embeddings_data):
        video_title = row.get('title', f'Video {i+1}')
        start_time = int(row.get('start', 0))
        end_time = int(row.get('end', 0))
        text_content = row.get('text', 'No content available')
        
        # Format time as MM:SS
        start_formatted = f"{start_time//60}:{start_time%60:02d}"
        end_formatted = f"{end_time//60}:{end_time%60:02d}"
        
        response_parts.append(f"📹 {video_title}")
        response_parts.append(f"   Time: {start_formatted} - {end_formatted}")
        response_parts.append(f"   Content: {text_content}")
        response_parts.append("")
    
    response_parts.append("Note: This is a simplified response. For a more detailed answer, please ensure the generation API is working properly.")
    
    return "\n".join(response_parts)

def process_query(incoming_query):
    """Process a single query and return response."""
    global df
    
    if df is None:
        raise Exception("Embeddings not loaded")
    
    print(f"Processing query: {incoming_query}")
    
    # Create embedding for the question
    question_embedding = create_embedding([incoming_query])[0] 
    
    # Find similarities
    embeddings_array = np.vstack(df['embedding'])
    question_embedding_array = np.array([question_embedding])
    
    # Normalize embeddings to prevent overflow/underflow issues
    embeddings_array = embeddings_array / (np.linalg.norm(embeddings_array, axis=1, keepdims=True) + 1e-8)
    question_embedding_array = question_embedding_array / (np.linalg.norm(question_embedding_array, axis=1, keepdims=True) + 1e-8)
    
    similarities = cosine_similarity(embeddings_array, question_embedding_array).flatten()
    
    top_results = 5
    max_indx = similarities.argsort()[::-1][0:top_results]
    new_df = df.iloc[max_indx] 
    
    # Create prompt
    prompt = f'''I am teaching Mathematics in my Math Class course. Here are video subtitle chunks containing video title, video number, start time in seconds, end time in seconds, the text at that time:

{new_df[["title", "number", "start", "end", "text"]].to_json(orient="records")}
---------------------------------
"{incoming_query}"
User asked this question related to the video chunks, you have to answer in a human way (dont mention the above format, its just for you) where and how much content is taught in which video (in which video and at what timestamp) and guide the user to go to that particular video. If user asks unrelated question, tell him that you can only answer questions related to the course
'''
    
    # Generate response
    try:
        response_data = inference(prompt)
        response = response_data["response"]
    except Exception as e:
        print(f"API error: {e}")
        # Fallback response when API times out
        response = create_fallback_response(new_df.to_dict('records'), incoming_query)
    
    return response

# Routes
@app.route('/')
def index():
    """Serve the main page."""
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat messages."""
    try:
        data = request.get_json()
        message = data.get('message', '').strip()
        
        if not message:
            return jsonify({'error': 'Message cannot be empty'}), 400
        
        # Process the query
        response = process_query(message)
        
        return jsonify({'response': response})
        
    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/health')
def health():
    """Health check endpoint."""
    return jsonify({'status': 'healthy', 'embeddings_loaded': df is not None})

if __name__ == '__main__':
    try:
        # Load embeddings on startup
        load_embeddings()
        print("Starting Flask app...")
        app.run(debug=True, host='0.0.0.0', port=8080)
    except Exception as e:
        print(f"Failed to start app: {e}")
        sys.exit(1)
