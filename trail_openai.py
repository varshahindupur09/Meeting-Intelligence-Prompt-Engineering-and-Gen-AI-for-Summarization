# from openai import OpenAI
# import openai

# openai.api_key = "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"
# client = OpenAI()

# audio_file = open("/Users/varshahindupur/Downloads/audio2_trim.mp3", "rb")
# transcript = client.audio.transcriptions.create(
#   file=audio_file,
#   model="whisper-1",
#   response_format="verbose_json",
#   timestamp_granularities=["word"]
# )

# print(transcript.words)

# from fastapi import FastAPI, HTTPException, UploadFile, File
# import os
# import openai
# import uvicorn

# app = FastAPI()

# # Set your OpenAI API key
# openai.api_key = "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"

# client = openai

# @app.post("/transcribe/")
# async def transcribe_audio(file: UploadFile = File(...)):
#     file_location = f"temp_{file.filename}"
#     try:
#         # Save the uploaded file temporarily
#         with open(file_location, "wb") as buffer:
#             buffer.write(await file.read())

#         # Open the file and transcribe using the OpenAI API
#         with open(file_location, "rb") as audio_file:
#             transcript = openai.Audio.transcriptions.create(
#                 file=audio_file,
#                 model="whisper-1",
#                 response_format="verbose_json",
#                 timestamp_granularities=["word"]
#             )
        
#         # Return the words with timestamps from the transcript
#         return {"words": transcript["words"]}

#     except Exception as e:
#         # Log the error and return an HTTP 500 response
#         print(f"Error during transcription: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")
    
#     finally:
#         # Clean up the temporary file
#         if os.path.exists(file_location):
#             os.remove(file_location)

# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)


from fastapi import FastAPI, HTTPException, UploadFile, File
import os
from openai import OpenAI
import uvicorn

app = FastAPI()

# Initialize OpenAI client with API key from environment variables
client = OpenAI(
    api_key="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"
)

@app.post("/transcribe/")
async def transcribe_audio(file: UploadFile = File(...)):
    file_location = f"temp_{file.filename}"
    try:
        # Save the uploaded file temporarily
        with open(file_location, "wb") as buffer:
            buffer.write(await file.read())

        # Open the file and transcribe using the OpenAI client
        with open(file_location, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                file=audio_file,
                model="whisper-1",
                response_format="verbose_json"
            )
        
        # Return the full transcription text as a paragraph
        return {"transcription": transcript.text}

    except Exception as e:
        # Log the error and return an HTTP 500 response
        print(f"Error during transcription: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")
    
    finally:
        # Clean up the temporary file
        if os.path.exists(file_location):
            os.remove(file_location)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)


    # @app.post("/transcribe/")
# async def transcribe_audio(file: UploadFile = File(...)):
#     file_location = f"temp_{file.filename}"
#     try:
#         # Save the uploaded file temporarily
#         with open(file_location, "wb") as buffer:
#             buffer.write(await file.read())

#         # Open the file and transcribe using the OpenAI client
#         with open(file_location, "rb") as audio_file:
#             transcript = client_openai.audio.transcriptions.create(
#                 file=audio_file,
#                 model="whisper-1",
#                 response_format="verbose_json"
#             )
        
#         # Return the full transcription text as a paragraph
#         return {"transcription": transcript.text}

#     except Exception as e:
#         # Log the error and return an HTTP 500 response
#         print(f"Error during transcription: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")
    
#     finally:
#         # Clean up the temporary file
#         if os.path.exists(file_location):
#             os.remove(file_location)