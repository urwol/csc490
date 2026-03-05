from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
import os
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

# Load environment variables (.env) to access API key
load_dotenv()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OpenAI client (gets API key from .env file for security)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM_INSTRUCTIONS = """
You are a research assistant chatbot for the Rider University Library.

Your job is to help students:
- find academic sources based on keywords, ideas, or rough drafts
- generate citations and references depending on desired format (ex. MLA, APA, Chicago, etc.)
- refine research topics and recommend alternate ideas based on research topics

You are NOT to write the students' papers for them, just aid in assisting them find sources and create citations/references.
Keep responses clear, helpful, and concise for students doing research. 
If urged to write a students introduction, conclusion, body paragraph, or anything of similar nature, ONLY provide them with tips for doing so, NOT examples or example structure.


Format responses clearly using bullet points, numbered lists, and line breaks.
Separate each source clearly.
"""

# Store conversation history
conversation = [
    {
        "role": "system",
        "content": [{"type": "input_text", "text": SYSTEM_INSTRUCTIONS}],
    }
]

# Define request format
class Message(BaseModel):
    message: str

@app.get("/")
def home():
    return FileResponse("index.html")

@app.post("/chat")
def chat(msg: Message):
    user_text = msg.message

    # Adds user message to conversation
    conversation.append(
        {"role": "user",
         "content": [{"type": "input_text", "text": user_text}]}
    )

    # Call OpenAI and generate response
    response = client.responses.create(
        model="gpt-4o-mini",
        input=conversation # type: ignore
    )

    assistant_text = response.output_text

    # Take the reply and save it and send to front end
    conversation.append(
        {"role": "assistant",
         "content": [{"type": "output_text", "text": assistant_text}]}
    )

    return {"reply": assistant_text}