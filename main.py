import os
import json
import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

class JobPrompt(BaseModel):
    text: str

class RawFixPrompt(BaseModel):
    raw_text: str

@app.get("/")
async def root():
    return {"message": "✅ Job Extraction API is running with Groq LLM"}

@app.api_route("/ping", methods=["GET", "HEAD"])
async def ping(request: Request):
    if request.method == "HEAD":
        return JSONResponse(content=None, status_code=200)
    return {"status": "ok"}

@app.post("/fix-json")
async def fix_raw_json(data: RawFixPrompt):
    prompt = f"""
You will receive a text that looks like JSON but may be invalid.

Fix it and return a valid JSON object. Each job should be inside a "jobs" array and have these fields:
company, role, batch, link, location, stipend, salary, duration, mode, other_info.

If any field is missing, use null. Ensure it's valid JSON. Do not return any explanations or markdown.

Text:
{data.raw_text}
"""

    completion = client.chat.completions.create(
        model="gemma2-9b-it",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
        max_tokens=1024,
        top_p=0.9,
        response_format={"type": "json_object"},
    )

    answer = completion.choices[0].message.content.strip()

    try:
        parsed = json.loads(answer)
        return {"result": parsed}
    except json.JSONDecodeError:
        return {
            "error": "Failed to parse into valid JSON",
            "raw_response": answer
        }

@app.post("/extract-job")
async def extract_job(data: JobPrompt):
    prompt = f"""
You are a job post extractor.

Instructions:
- Extract structured job data only if the input has job/internship info and a valid apply link (not telegram/youtube/whatsapp).
- Return a JSON object: {{"jobs": [{{job1}}, {{job2}}, ...]}}
- Use only these keys in each job: company, role, batch, link, location, stipend, salary, duration, mode, other_info.
- If any field is missing, set it to null.
- If no job is found or no valid links, return `null` (without quotes).
- Do NOT include any markdown, extra commentary, or invalid formatting.
- Output must be syntactically valid JSON.

Text:
{data.text}
"""

    completion = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=1024,
        response_format={"type": "json_object"},
        top_p=0.9,
    )

    answer = completion.choices[0].message.content.strip()

    # Handle exact "null" return
    if answer.strip().lower() == "null":
        return {"result": None}

    # Try parsing directly
    try:
        parsed = json.loads(answer)
        return {"result": parsed}
    except json.JSONDecodeError:
        # Fallback to /fix-json route
        try:
            async with httpx.AsyncClient() as xclient:
                fix_resp = await xclient.post(
                    "https://llm-59ws.onrender.com/fix-json",
                    json={"raw_text": answer}
                )
                fix_json = fix_resp.json()
                return {"result": fix_json.get("result")}
        except Exception as fallback_error:
            return {
                "error": "Failed to parse LLM output and fallback also failed.",
                "raw_llm": answer,
                "fallback_error": str(fallback_error)
            }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
