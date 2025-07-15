from pytube import YouTube
import os
import whisper
from youtube_transcript_api import YouTubeTranscriptApi
import streamlit as st
from pydub import AudioSegment  # Required for format conversion

def transcribe_with_whisper(url):
    try:
        # Set up paths
        output_dir = os.path.join(os.getcwd(), "Youtube")
        os.makedirs(output_dir, exist_ok=True)
        audio_filename = "audio.mp4"
        audio_path = os.path.join(output_dir, audio_filename)

        # Step 1: Download audio
        yt = YouTube(url)
        audio_stream = yt.streams.filter(only_audio=True, file_extension="mp4").first()
        print("Downloading audio...")
        audio_stream.download(output_path=output_dir, filename=audio_filename)

        # Optional: Convert to WAV (more compatible)
        wav_path = os.path.join(output_dir, "audio.wav")
        print("Converting to WAV...")
        sound = AudioSegment.from_file(audio_path)
        sound.export(wav_path, format="wav")

        # Step 2: Load Whisper model
        print("Transcribing using Whisper...")
        model = whisper.load_model("base")  # or "small", "medium", "large"
        result = model.transcribe(wav_path)

        # Step 3: Save transcript
        transcribed_text = result['text']
        output_file_path = os.path.join(output_dir, "video.txt")
        with open(output_file_path, 'w', encoding='utf-8') as output_file:
            output_file.write(transcribed_text)

        print("Transcription complete. Saved to:", output_file_path)
        return transcribed_text

    except Exception as e:
        print(f"Whisper failed: {e}")
        st.warning("Whisper failed...", icon="⚠️")
        return None

def transcribe_video_to_text(url):
    try:
        yt = YouTube(url)
        # video_title = yt.title
        video_id = url.split("=")[-1]
        srt1 = YouTubeTranscriptApi.get_transcript(video_id)
        transcript_text = " ".join([i['text'] for i in srt1])
        output_dir = os.path.join(os.getcwd(), "Youtube")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_file_path = os.path.join(output_dir, "video.txt")
        with open(output_file_path, 'w', encoding='utf-8') as output_file:
            output_file.write(transcript_text)
        return transcript_text
    except Exception as e:
        # Fall back to Whisper if transcript not available
        st.warning("Transcript not found. Using Whisper instead...", icon="⚠️")
        print(f"Transcript API failed: {e}")
        # return transcribe_with_whisper(url)

# Example usage:
if __name__ == "__main__":
    video_url = "https://www.youtube.com/watch?v=quCEmM2JBbk"
    transcribe_video_to_text(video_url)




