# RAG-based AI Learning Assistant

This system uses Retrieval-Augmented Generation (RAG) to answer questions about educational video content. It processes MP4 videos, creates embeddings, and provides intelligent responses based on the video content.

## Prerequisites

1. **OpenAI API Key**: Get your API key from https://platform.openai.com/api-keys
   - Create a `.env` file in the project root
   - Add your API key: `OPENAI_API_KEY=your_api_key_here`

2. **Python Dependencies**: Install required packages
   ```bash
   pip install -r requirements.txt
   ```

## Setup Instructions

1. **Prepare Video Files**: Place your MP4 educational videos in the `learning_videos/` directory

2. **Process Videos**: Convert MP4 files to JSON transcripts
   ```bash
   python mp4_to_json.py
   ```

3. **Create Embeddings**: Generate embeddings for the video content
   ```bash
   python preprocess.py
   ```

4. **Ask Questions**: Start the interactive Q&A system
   ```bash
   python process_incoming.py
   ```

## File Structure

- `mp4_to_json.py`: Converts MP4 videos to JSON transcripts using Whisper
- `preprocess.py`: Creates embeddings from video transcripts
- `process_incoming.py`: Main Q&A interface
- `learning_videos/`: Directory containing MP4 video files
- `jsons/`: Directory containing JSON transcript files
- `embeddings.joblib`: Precomputed embeddings for fast retrieval
- `requirements.txt`: Python dependencies

## Usage

1. Run the system in the correct order:
   ```bash
   # Step 1: Convert videos to transcripts
   python mp4_to_json.py
   
   # Step 2: Create embeddings
   python preprocess.py
   
   # Step 3: Ask questions
   python process_incoming.py
   ```

2. The system will prompt you to ask questions about the video content
3. It will provide relevant video segments and timestamps for your questions

## Configuration

You can customize the system by editing the `.env` file:

```env
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_CHAT_MODEL=gpt-3.5-turbo
OPENAI_MAX_TOKENS=1000
OPENAI_TEMPERATURE=0.7
```

## Troubleshooting

- **OpenAI API errors**: Check your API key and billing status
- **NumPy compatibility issues**: Make sure you have NumPy < 2.0 installed
- **File not found errors**: Check that you've run the preprocessing steps in order
- **Memory issues**: For large video files, consider using smaller Whisper models

## Notes

- The system uses the `large-v2` Whisper model by default for transcription
- Embeddings are created using OpenAI's `text-embedding-3-small` model
- Responses are generated using OpenAI's `gpt-3.5-turbo` model
- All API calls include proper error handling and retry logic
