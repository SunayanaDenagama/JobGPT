import os
import json
import time
from pydantic import BaseModel, Field
from supabase import create_client, Client
from google import genai
from google.genai import types
from google.genai.errors import APIError
from dotenv import load_dotenv

load_dotenv()

supabase: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_KEY")
)
ai_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Strict output schema
class JobMatch(BaseModel):
    ref_no: str
    title: str
    company: str
    apply_url: str
    match_score: int = Field(description="Score between 0 and 100 based on candidate fit")
    fit_reason: str = Field(description="2 sentences explaining why this role matches the candidate profile")
    skill_gaps: str = Field(description="Any skills mentioned in the role that the candidate might lack")

class MatchReport(BaseModel):
    top_matches: list[JobMatch]

def evaluate_jobs_for_candidate(profile_description: str, target_location: str = "Any"):
    print("[*] Reading latest active postings from Supabase...")
    
    db_response = supabase.table("jobs").select("ref_no, title, company, location, apply_url").order("created_at", desc=True).limit(40).execute()
    jobs_data = db_response.data

    if not jobs_data:
        print("[!] No jobs found in database. Run sync_jobs.py first.")
        return None

    # Using the verified latest production endpoint
    model_name = "gemini-flash-latest"
    print(f"[*] Analyzing {len(jobs_data)} jobs against your profile using {model_name}...")

    prompt = f"""
    You are an expert career mentor and technical recruiter in Sri Lanka.
    
    Candidate Profile & Preferences:
    - Background: {profile_description}
    - Location Preference: {target_location}
    
    Available Job Postings:
    {json.dumps(jobs_data, indent=2)}
    
    Task:
    1. Filter and identify the top 3-5 best matching opportunities for this candidate.
    2. Score each out of 100 based on realistic technical fit.
    3. State clearly why it's a fit and note any missing skills or experience gaps.
    """

    for attempt in range(3):
        try:
            response = ai_client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=MatchReport,
                    temperature=0.1,
                ),
            )
            report = MatchReport.model_validate_json(response.text)
            return report

        except APIError as e:
            if e.code == 503 and attempt < 2:
                wait_time = (attempt + 1) * 3
                print(f"[!] Endpoint busy (503). Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise e

if __name__ == "__main__":
    my_profile = "Junior developer with skills in Python, basic React, Git, and SQL. Looking for entry-level, associate, or trainee software engineering roles."
    
    report = evaluate_jobs_for_candidate(my_profile, target_location="Colombo")

    if report:
        print("\n================ TOP MATCHED JOBS ================\n")
        for match in report.top_matches:
            print(f"📌 {match.title} at {match.company}")
            print(f"   Match Score: {match.match_score}%")
            print(f"   Why it fits: {match.fit_reason}")
            print(f"   Potential Gaps: {match.skill_gaps}")
            print(f"   Apply Link: {match.apply_url}")
            print("-" * 50)