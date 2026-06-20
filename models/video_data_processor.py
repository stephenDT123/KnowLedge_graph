import moviepy.editor as mp
import speech_recognition as sr

def extract_audio_from_video(video_path):
    video = mp.VideoFileClip(video_path)
    audio_path = "temp_audio.wav"
    video.audio.write_audiofile(audio_path)
    return audio_path

def transcribe_audio(audio_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio)
        except sr.UnknownValueError:
            text = "无法识别音频"
    return text

# 示例使用
if __name__ == "__main__":
    video_path = "example.mp4"
    audio_path = extract_audio_from_video(video_path)
    transcribed_text = transcribe_audio(audio_path)
    print(transcribed_text)