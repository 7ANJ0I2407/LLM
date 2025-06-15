from fastapi import FastAPI
from pydantic import BaseModel
from gradio_client import Client

app = FastAPI()
client = Client("hysts/mistral-7b")

class JobPrompt(BaseModel):
    text: str

@app.get("/")
async def root():
    return {"message": "✅ Job Extraction API is running!"}

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
    result = client.predict(
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
