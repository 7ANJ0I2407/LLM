# main.py
import os
from fastapi import FastAPI
from pydantic import BaseModel
from gradio_client import Client
import uvicorn
from fastapi import Request
from fastapi.responses import JSONResponse
from fastapi.concurrency import run_in_threadpool


app = FastAPI()
client = Client("hysts/mistral-7b")

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
You are an intelligent job description parser.

Given the following text, do one of two things:
- If it's a job/internship/career-related announcement, return structured details (e.g. company, role, batch, link, location, stipend/salary, duration, mode, other info).
- If it's not a job-related text, respond with: "null"

Text: {data.text}
[/INST]
"""
    try:
        result = await run_in_threadpool(
            client.predict,
            message=prompt,
            param_2=1024,
            param_3=0.6,
            param_4=0.9,
            param_5=50,
            param_6=1.2,
            api_name="/chat"
        )

        answer = result.split("[/INST]")[-1].strip()
        if answer.lower().strip() == "null":
            return {"result": None}
        return {"result": answer}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# This ensures the correct port is used on Render
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
