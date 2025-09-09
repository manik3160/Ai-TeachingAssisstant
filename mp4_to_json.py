import whisper
import json
import os
import sys


def main():
    """Main function to transcribe MP4 videos and save as JSON."""
    try:
        # Check if learning_videos directory exists
        if not os.path.exists("learning_videos"):
            print("Error: 'learning_videos' directory not found.")
            print("Please create the directory and add your MP4 files to it.")
            sys.exit(1)
            
        print("Loading Whisper model...")
        model = whisper.load_model("large-v2")
        print("Model loaded successfully")

        # Create output directory
        os.makedirs("jsons", exist_ok=True)

        # Get list of MP4 files
        audios = os.listdir("learning_videos")
        mp4_files = [f for f in audios if f.endswith(".mp4")]
        
        if not mp4_files:
            print("Error: No MP4 files found in 'learning_videos' directory.")
            sys.exit(1)
            
        print(f"Found {len(mp4_files)} MP4 files to process")

        for audio in mp4_files:
            try:
                print(f"Transcribing: {audio}")

                # Transcribe audio
                result = model.transcribe(
                    audio=f"learning_videos/{audio}",
                    language="en",
                    word_timestamps=False
                )

                # Create chunks
                chunks = []
                for i, segment in enumerate(result["segments"], start=1):
                    chunks.append({
                        "number": i,
                        "title": audio.replace(".mp4", ""),
                        "start": segment["start"],
                        "end": segment["end"],
                        "text": segment["text"]
                    })

                # Create metadata
                chunks_with_metadata = {
                    "file": audio,
                    "chunks": chunks,
                    "full_text": result["text"]
                }

                # Save to JSON
                output_file = f"jsons/{audio.replace('.mp4', '.json')}"
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(chunks_with_metadata, f, ensure_ascii=False, indent=2)

                print(f"✅ Saved: {output_file}")
                
            except Exception as e:
                print(f"Error processing {audio}: {e}")
                continue
                
        print(f"\nCompleted processing {len(mp4_files)} files")
        
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(0)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
