import pandas as pd 
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np 
import joblib 
import os
import sys
import time
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))


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
                print("All retry attempts failed. Please check your OpenAI API key and try again.")
                sys.exit(1)
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
                print("All retry attempts failed. Please check your OpenAI API key and try again.")
                sys.exit(1)
            time.sleep(2)  # Wait before retry

def create_fallback_response(df, query):
    """Create a simple fallback response when the generation API is unavailable."""
    response_parts = []
    response_parts.append(f"Based on your question '{query}', I found the following relevant video content:\n")
    
    for idx, row in df.iterrows():
        video_title = row['title']
        start_time = int(row['start'])
        end_time = int(row['end'])
        text_content = row['text']
        
        # Format time as MM:SS
        start_formatted = f"{start_time//60}:{start_time%60:02d}"
        end_formatted = f"{end_time//60}:{end_time%60:02d}"
        
        response_parts.append(f"{video_title}")
        response_parts.append(f"   Time: {start_formatted} - {end_formatted}")
        response_parts.append(f"   Content: {text_content}")
        response_parts.append("")
    
    response_parts.append("Note: This is a simplified response. For a more detailed answer, please ensure the generation API is working properly.")
    
    return "\n".join(response_parts)

def main():
    """Main function to process incoming queries and generate responses."""
    try:
        # Load embeddings
        if not os.path.exists('embeddings.joblib'):
            print("Error: embeddings.joblib file not found. Please run preprocess.py first.")
            sys.exit(1)
            
        print("Loading embeddings...")
        df = joblib.load('embeddings.joblib')
        print(f"Loaded {len(df)} embeddings successfully")
        print("\n" + "="*60)
        print(" RAG-based AI Math Tutor")
        print("Ask me anything about geometry and transformations!")
        print("Type 'bye' to exit the chat.")
        print("="*60 + "\n")
        
       
        while True:
      
            incoming_query = input("You: ").strip()
    
            if incoming_query.lower() in ['bye', 'exit', 'quit', 'goodbye']:
                print("\n AI Tutor: Goodbye! Thanks for learning with me! 👋")
                break
                
            if not incoming_query:
                print("Please ask a question or type 'bye' to exit.")
                continue
            
            print("Creating embedding for your question...")
            question_embedding = create_embedding([incoming_query])[0] 
            
        
            print("Finding similar content...")
            
   
            embeddings_array = np.vstack(df['embedding'])
            question_embedding_array = np.array([question_embedding])
            
            
            embeddings_array = embeddings_array / (np.linalg.norm(embeddings_array, axis=1, keepdims=True) + 1e-8)
            question_embedding_array = question_embedding_array / (np.linalg.norm(question_embedding_array, axis=1, keepdims=True) + 1e-8)
            
            similarities = cosine_similarity(embeddings_array, question_embedding_array).flatten()
            
            top_results = 5
            max_indx = similarities.argsort()[::-1][0:top_results]
            new_df = df.loc[max_indx] 
            
            
            prompt = f'''I am teaching Mathematics in my Math Class course. Here are video subtitle chunks containing video title, video number, start time in seconds, end time in seconds, the text at that time:

{new_df[["title", "number", "start", "end", "text"]].to_json(orient="records")}
---------------------------------
"{incoming_query}"
User asked this question related to the video chunks, you have to answer in a human way (dont mention the above format, its just for you) where and how much content is taught in which video (in which video and at what timestamp) and guide the user to go to that particular video. If user asks unrelated question, tell him that you can only answer questions related to the course
'''
            
            # Save prompt
            try:
                with open("prompt.txt", "w", encoding="utf-8") as f:
                    f.write(prompt)
                print("Prompt saved to prompt.txt")
            except IOError as e:
                print(f"Warning: Could not save prompt to file: {e}")
            
            # Generate response
            print("Generating response...")
            try:
                response_data = inference(prompt)
                response = response_data["response"]
            except SystemExit:
                # Fallback response when API times out
                print("API timeout - providing fallback response based on similar content...")
                response = create_fallback_response(new_df, incoming_query)
            
            print("\n" + "="*50)
            print("🤖 AI Tutor:")
            print("="*50)
            print(response)
            print("="*50)
            
            # Save response
            try:
                with open("response.txt", "w", encoding="utf-8") as f:
                    f.write(response)
                print("\nResponse saved to response.txt")
            except IOError as e:
                print(f"Warning: Could not save response to file: {e}")
            
            print("\n" + "-"*60 + "\n")
            
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(0)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()