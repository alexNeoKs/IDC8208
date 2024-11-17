from dotenv import load_dotenv
import os

load_dotenv()  # This loads the environment variables from .env
print(os.environ["IMAGEIO_FFMPEG_EXE"])  # Verify the value
