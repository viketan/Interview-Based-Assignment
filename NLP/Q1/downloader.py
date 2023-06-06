import csv
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from oauth2client.tools import argparser

# Set up API credentials
DEVELOPER_KEY = "AIzaSyAn9gCI3iVSanYxTuVml9ahFr16FCIZimk"
YOUTUBE_API_SERVICE_NAME = "youtube"
YOUTUBE_API_VERSION = "v3"

# Function to retrieve comments from a video
def get_video_comments(video_id):
    youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=DEVELOPER_KEY)

    try:
        # Retrieve comments using the YouTube Data API
        response = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            textFormat="plainText",
            maxResults=100,
        ).execute()

        comments = []

        # Parse API response and extract comments
        while response:
            for item in response["items"]:
                comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
                comments.append(comment)

            # Fetch next page of comments, if available
            if "nextPageToken" in response:
                response = youtube.commentThreads().list(
                    part="snippet",
                    videoId=video_id,
                    textFormat="plainText",
                    maxResults=100,
                    pageToken=response["nextPageToken"]
                ).execute()
            else:
                break

        return comments

    except HttpError as e:
        print("An HTTP error %d occurred:\n%s" % (e.resp.status, e.content))
        return []

# Main function to run the code
def main():
    
    video_id = input("Please Enter Youtube video id: ")

    # Retrieve comments from the video
    comments = get_video_comments(video_id)

    if comments:
        # Store comments in a CSV file
        with open("youtube_comments.csv", "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Comment"])
            for comment in comments:
                writer.writerow([comment])

        print("Comments extracted and saved in 'youtube_comments.csv'.")
    else:
        print("No comments found for the given video.")

# Execute the main function
if __name__ == "__main__":
    main()
