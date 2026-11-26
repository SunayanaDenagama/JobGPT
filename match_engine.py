import os
import json
from pydantic import BaseModel, Field
from supabase import create_client, Client
from dotenv import load_dotenv
from gemini_service import GeminiGenerator

load_dotenv()

supabase: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_KEY")
)
generator = GeminiGenerator.from_key(os.environ["GEMINI_API_KEY"], os.getenv("GEMINI_MODEL", ""))

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

    print(f"[*] Analyzing {len(jobs_data)} jobs against your profile...")

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

    report, error = generator.generate(prompt, MatchReport)
    if error:
        print(f"[!] {error}")
    return report

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
