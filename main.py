from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Response
from fastapi.middleware.cors import CORSMiddleware
import json
from pydantic import BaseModel
from typing import Optional
import PyPDF2
import io

import os
from dotenv import load_dotenv

from GeminiClient import GeminiClient
from prompts import prompt

load_dotenv()

app = FastAPI()

# Update CORS to allow your Vercel deployment URL
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins or specify your Vercel URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Article(BaseModel):
    article: Optional[str] = None

@app.get("/")
def read_root():
    return {"message": "The Server is Running!"}


@app.post("/summarize")
async def summarize(request: Request, file: UploadFile = File(None)):
    text = None
    try:
        if file:
            print(f"Processing file: {file.filename}")
            if file.content_type != 'application/pdf':
                raise HTTPException(status_code=400, detail="Invalid file type")

            contents = await file.read()
            pdf_file = PyPDF2.PdfReader(io.BytesIO(contents))
            text = ""
            for page in pdf_file.pages:
                try:
                    text += page.extract_text()
                except Exception as e:
                    print(f"Error extracting text from page: {e}")
                    continue  # Skip to the next page if there's an error
            
            print(f"Extracted {len(text)} characters from PDF")

        else:
            body = await request.body()
            try:
                data = json.loads(body.decode())
                article = data.get("article")
                if article:
                    text = article
                    print(f"Received article text: {len(text)} characters")
                else:
                    raise HTTPException(
                        status_code=400, detail="No article or file provided"
                    )
            except (json.JSONDecodeError, AttributeError) as e:
                raise HTTPException(
                    status_code=400, detail=f"Invalid JSON data: {e}"
                )

    except PyPDF2.errors.PdfReadError as e:
        print(f"PDF read error: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid PDF file: {e}")
    except Exception as e:
        print(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing request: {e}")

    if not text or len(text.strip()) < 50:  # Ensure we have meaningful text
        print("Text is too short or empty")
        raise HTTPException(status_code=400, detail="Text is too short or empty")

    # Check for both environment variable names
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="API key not set. Check environment variables.")

    gemini_client = GeminiClient(api_key)
    try:
        print("Generating summary...")
        final_prompt = prompt.format(article=text)
        try:
            summary = gemini_client.generate_content(final_prompt)
            print(f"Summary generated: {len(summary)} characters")
        except Exception as e:
            print(f"Error generating summary: {e}")
            raise HTTPException(status_code=500, detail=f"Error generating summary: {e}")
        
        print("Generating flashcards...")
        try:
            flashcards = gemini_client.generate_flashcards(summary)
            num_flashcards = len(flashcards) if isinstance(flashcards, list) else 0
            print(f"Generated {num_flashcards} flashcards")
            
            # Ensure we're returning reasonable number of flashcards
            if num_flashcards < 3 and len(summary) > 500:
                print("Too few flashcards, trying again with different prompt...")
                # Try again with a different approach
                flashcards = gemini_client.generate_flashcards(
                    f"Create at least 5 detailed flashcards from this text:\n{summary}"
                )
                num_flashcards = len(flashcards) if isinstance(flashcards, list) else 0
                print(f"Second attempt generated {num_flashcards} flashcards")
            
            if not flashcards or num_flashcards == 0:
                raise HTTPException(status_code=500, detail="Failed to generate flashcards")
                
        except Exception as e:
            print(f"Error generating flashcards: {e}")
            raise HTTPException(status_code=500, detail=f"Error generating flashcards: {e}")
        
        return {"flashcards": flashcards}
    except Exception as e:
        print(f"Error in final processing: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
#to check if the server are able to receive data and return it back
@app.post("/echo")
async def echo_data(request: Request):
    data = await request.json()
    return data

@app.post("/read-pdf")
async def read_pdf(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Invalid file type. Only PDF allowed.")

    try:
        contents = await file.read()
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(contents))
        text = ""

        for page in pdf_reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted

        if not text.strip():
            raise HTTPException(status_code=400, detail="No text found in PDF.")
        
        return {"extracted_text": text.strip()}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading PDF: {e}")

# This is the handler that Vercel will use
from mangum import Mangum
handler = Mangum(app)
