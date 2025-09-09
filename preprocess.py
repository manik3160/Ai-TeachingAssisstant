import os
import json
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import joblib
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
            print(f"Creating embeddings (attempt {attempt + 1}/{max_retries})...")
            
            # Use OpenAI embeddings API
            response = client.embeddings.create(
                model=os.getenv('OPENAI_EMBEDDING_MODEL', 'text-embedding-3-small'),
                input=text_list
            )
            
            # Extract embeddings from response
            embeddings = [data.embedding for data in response.data]
            print("Embeddings created successfully")
            return embeddings
            
        except Exception as e:
            print(f"Error creating embeddings (attempt {attempt + 1}): {e}")
            if attempt == max_retries - 1:
                print("All retry attempts failed. Please check your OpenAI API key and try again.")
                sys.exit(1)
            time.sleep(2)  # Wait before retry


def main():
    """Main function to process JSON files and create embeddings."""
    try:
        # Check if jsons directory exists
        if not os.path.exists("jsons"):
            print("Error: 'jsons' directory not found. Please run mp4_to_json.py first.")
            sys.exit(1)
            
        jsons = os.listdir("jsons")  # List all the jsons
        
        if not jsons:
            print("Error: No JSON files found in 'jsons' directory.")
            sys.exit(1)
            
        print(f"Found {len(jsons)} JSON files to process")
        
        my_dicts = []
        chunk_id = 0

        for json_file in jsons:
            if not json_file.endswith('.json'):
                print(f"Skipping non-JSON file: {json_file}")
                continue
                
            try:
                with open(f"jsons/{json_file}", 'r', encoding='utf-8') as f:
                    content = json.load(f)
                    
                if 'chunks' not in content:
                    print(f"Warning: No 'chunks' found in {json_file}, skipping...")
                    continue
                    
                print(f"Creating Embeddings for {json_file}")
                embeddings = create_embedding([c['text'] for c in content['chunks']])
                   
                for i, chunk in enumerate(content['chunks']):
                    chunk['chunk_id'] = chunk_id
                    chunk['embedding'] = embeddings[i]
                    chunk_id += 1
                    my_dicts.append(chunk)
                    
            except json.JSONDecodeError as e:
                print(f"Error: Invalid JSON in {json_file}: {e}")
                continue
            except IOError as e:
                print(f"Error reading {json_file}: {e}")
                continue
            except Exception as e:
                print(f"Error processing {json_file}: {e}")
                continue

        if not my_dicts:
            print("Error: No valid chunks found to process.")
            sys.exit(1)
            
        print(f"Creating DataFrame with {len(my_dicts)} chunks...")
        df = pd.DataFrame.from_records(my_dicts)
        
        # Save this dataframe 
        print("Saving embeddings to embeddings.joblib...")
        joblib.dump(df, "embeddings.joblib")
        print(f"Successfully saved {len(df)} embeddings to embeddings.joblib")
        
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(0)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
