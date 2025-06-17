import os
import json
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from gradio_client import Client
import uvicorn

# Optional: Uncomment if you want to keep your HF token secure
HF_TOKEN = os.getenv("HF_TOKEN")
client = Client("hysts/mistral-7b", hf_token=HF_TOKEN)

# client = Client("hysts/mistral-7b")
app = FastAPI()

class JobPrompt(BaseModel):
    text: str

@app.get("/")
async def root():
    return {"message": "✅ Job Extraction API is running!"}

@app.api_route("/ping", methods=["GET", "HEAD"])
async def ping(request: Request):
    if request.method == "HEAD":
        return JSONResponse(content=None, status_code=200)
    return {"status": "ok"}

@app.post("/extract-job")
async def extract_job(data: JobPrompt):
    prompt = f"""
[INST]
You are a smart job parser.

If the input text contains job/internship information, extract all relevant fields and return a valid JSON object. The keys can vary — extract only what's present (e.g. company, role, batch/graduation, link, location, stipend/salary, duration, mode, other info).

Make sure:
- The response is valid JSON only (no explanation, no markdown)
- Do not escape underscores unnecessarily
- If a field is missing, use null
- If the input is not job-related, just return null (without quotes)
[/INST]

Only return:
- A valid JSON object if details are available
- Or `null` (without quotes) if not enough info

Do NOT return any extra explanation, apology, or markdown.


{data.text}
"""

    try:
        raw_result = await run_in_threadpool(
            client.predict,
            message=prompt,
            param_2=1024,
            param_3=0.6,
            param_4=0.9,
            param_5=50,
            param_6=1.2,
            api_name="/chat"
        )

        response_text = raw_result.strip()

        # Step 1: Handle non-job post
        if response_text.lower() == "null":
            return {"result": None}

        # Step 2: Clean known issues
        cleaned = response_text

        # Remove markdown if present
        if cleaned.startswith("```json") or cleaned.startswith("```"):
            cleaned = "\n".join(cleaned.split("\n")[1:-1])

        # Fix common formatting issues
        cleaned = cleaned.replace("\\_", "_")
        cleaned = cleaned.replace(",}", "}")
        cleaned = cleaned.replace(",]", "]")
        cleaned = cleaned.replace("“", '"').replace("”", '"')
        cleaned = cleaned.replace("‘", "'").replace("’", "'")

        # Strip leading explanations (grab only JSON part)
        json_start = cleaned.find("{")
        if json_start == -1:
            raise ValueError("No JSON found")
        cleaned = cleaned[json_start:]

        # Final parse
        parsed_json = json.loads(cleaned)
        if isinstance(parsed_json, dict):
            return {"result": parsed_json}
        else:
            return JSONResponse(status_code=422, content={"error": "Not a JSON object", "raw": response_text})

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e), "raw": str(raw_result)})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
