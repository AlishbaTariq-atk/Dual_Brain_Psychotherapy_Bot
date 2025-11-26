from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
import pandas as pd

df= pd.read_csv("urls.csv")
video_ids = df["Url"].apply(lambda x: x.split("=")[-1]).tolist()
df["Transcript"] = ""
try:
    transcripts = []
    for video_id in video_ids:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
        transcript = " ".join([entry["text"] for entry in transcript_list])
        transcripts.append(transcript)
        print(f"Transcript for video {video_id}:\n{transcript}")
except TranscriptsDisabled:
    print(f"Transcripts are disabled for video {video_id}.")

df["Transcript"] = transcripts
df.to_csv("transcripts.csv", index=False)