import whisper
from whisper.utils import format_timestamp

# Load the fast Whisper model
model = whisper.load_model("small")

# Path to your MP3 file
audio_path = r"C:\Users\jaspe\Downloads\WhatsApp-Audio-2025-06-25-at-10.05.09_ebcd0975.dat.mp3"

# Transcribe with Dutch language and visible progress
result = model.transcribe(audio_path, verbose=True)

# Save the output with timestamps
output_path = "transcription_output.txt"
with open(output_path, "w", encoding="utf-8") as f:
    for segment in result["segments"]:
        f.write(f"[{format_timestamp(segment['start'])} --> {format_timestamp(segment['end'])}] {segment['text']}\n")

print(f"\n✅ Fast transcription complete! Saved to {output_path}")
