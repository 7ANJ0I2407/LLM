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
You will receive a text that looks structured but is not valid JSON.

Convert it into valid JSON object with keys: company, role, batch, link, location, stipend, salary, duration, mode, other_info and others if given so.

If a field is not present, assign it as null.

Text:
{data.raw_text}
"""

    completion = client.chat.completions.create(
        model="gemma2-9b-it",
        messages=[{
            "role": "user",
            "content": prompt
        }],
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
You are a smart job parser.

If the input text contains job/internship information, extract all relevant fields and return a valid JSON object. The keys can vary — extract only what's present (e.g. company, role, batch/graduation, link, location, stipend/salary, duration, mode, other info).

Make sure:
- The response is valid JSON only (no explanation, no markdown)
- Do not escape underscores unnecessarily
- If a field is missing, use null
- If the input is not job-related, just return null (without quotes)

Only return:
- A valid JSON object if details are available
- Or null (without quotes) if not enough info

Text: {data.text}
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
    if answer.lower() == "null":
        return {"result": None}
    try:
        parsed = json.loads(answer)
        return {"result": parsed}
    except json.JSONDecodeError:
        async with httpx.AsyncClient() as xclient:
            fix_resp = await xclient.post("http://localhost:8000/fix-json", json={"raw_text": answer})
            return {"result": fix_resp.json()}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
