import os
import time
from dotenv import load_dotenv
from supabase import create_client, Client
from google import genai
from google.genai import types

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not SUPABASE_URL or not SUPABASE_KEY or not GEMINI_API_KEY:
    raise ValueError("Missing SUPABASE_URL, SUPABASE_KEY, or GEMINI_API_KEY in .env!")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
ai_client = genai.Client(api_key=GEMINI_API_KEY)

def backfill_embeddings():
    print("[*] Checking Supabase for listings lacking vector embeddings...")
    
    # Query only rows where embedding is null
    res = supabase.table("jobs").select("ref_no, title, company, location").is_("embedding", "null").execute()
    pending_jobs = res.data or []
    
    total = len(pending_jobs)
    if total == 0:
        print("[+] All jobs in your database are already 100% vectorized!")
        return

    print(f"[*] Found {total} unvectorized jobs. Beginning batch processing...")
    
    batch_size = 30
    total_batches = (total + batch_size - 1) // batch_size

    for i in range(0, total, batch_size):
        batch_num = (i // batch_size) + 1
        chunk = pending_jobs[i:i + batch_size]
        
        texts = [f"{j.get('title', '')} at {j.get('company', '')} in {j.get('location', '')}" for j in chunk]

        print(f"[*] Vectorizing batch {batch_num}/{total_batches} ({len(chunk)} records)...")

        # Retry loop for API backoff
        for attempt in range(4):
            try:
                resp = ai_client.models.embed_content(
                    model="gemini-embedding-001",
                    contents=texts,
                    config=types.EmbedContentConfig(output_dimensionality=768)
                )
                
                vectors = [emb.values for emb in resp.embeddings]

                # Update Supabase rows individually or in bulk
                for job, vec in zip(chunk, vectors):
                    supabase.table("jobs").update({"embedding": vec}).eq("ref_no", job["ref_no"]).execute()

                print(f"[+] Successfully saved batch {batch_num}/{total_batches}!")
                # Safe rate pacing between batches
                time.sleep(1.5)
                break

            except Exception as e:
                err_str = str(e)
                print(f"[!] Warning on attempt {attempt + 1}: {err_str}")
                if "RESOURCE_EXHAUSTED" in err_str or "429" in err_str:
                    print("[!] Rate threshold reached. Pausing 65 seconds...")
                    time.sleep(65)
                else:
                    time.sleep(3)
                    if attempt == 3:
                        print(f"[X] Skipping batch {batch_num} due to repeated error.")

    print("\n[+] Embedding pipeline finished! Validating database...")
    
    # Final count audit
    total_count = supabase.table("jobs").select("ref_no", count="exact").execute().count
    embedded_count = supabase.table("jobs").select("ref_no", count="exact").not_.is_("embedding", "null").execute().count
    print(f"[✔] Complete: {embedded_count} / {total_count} listings are now vectorized ({embedded_count / total_count * 100:.1f}%).")

if __name__ == "__main__":
    backfill_embeddings()