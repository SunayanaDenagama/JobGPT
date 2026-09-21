import os
import re
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

supabase: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_KEY")
)

print("[*] Fetching TopJobs records to repair URLs...")

# Fetch all columns or just primary keys needed
response = supabase.table("jobs").select("ref_no").eq("source", "TopJobs.lk").execute()
rows = response.data

print(f"[*] Found {len(rows)} TopJobs records to patch...")

success_count = 0
for i, row in enumerate(rows):
    ref_no = row["ref_no"]
    ref_id = re.sub(r'^[A-Za-z]+-', '', ref_no)
    new_url = f"https://www.topjobs.lk/emp/job/empjobattachement.jsp?ac=DEFZZZ&id={ref_id}"

    try:
        supabase.table("jobs").update({"apply_url": new_url}).eq("ref_no", ref_no).execute()
        success_count += 1
        if success_count % 100 == 0 or success_count == len(rows):
            print(f"    Updated {success_count}/{len(rows)} listings...")
    except Exception as e:
        print(f"    [!] Error on {ref_no}: {e}")

print(f"[+] Successfully repaired {success_count} TopJobs URLs!")