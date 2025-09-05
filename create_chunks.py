import whisper
import json
import os

model = whisper.load_model("large-v2")


os.makedirs("jsons", exist_ok=True)


audios = os.listdir("learning_videos")

for audio in audios:
    if audio.endswith(".mp4"):  # only process mp4 files
        print(f"Transcribing: {audio}")

        # Transcribe audio
        result = model.transcribe(
            audio=f"learning_videos/{audio}",
            language="en",
            word_timestamps=False
        )

        
        chunks = []
        for i, segment in enumerate(result["segments"], start=1):
            chunks.append({
                "number": i,
                "title": audio.replace(".mp4", ""),
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"]
            })

        
        chunks_with_metadata = {
            "file": audio,
            "chunks": chunks,
            "full_text": result["text"]
        }

        
        output_file = f"jsons/{audio.replace('.mp4', '.json')}"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(chunks_with_metadata, f, ensure_ascii=False, indent=2)

        print(f"✅ Saved: {output_file}")
