import os
import re
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


def build_enriched_embedding_text(title: str, company: str, location: str, source: str = "") -> str:
    """Enriches short job titles with implied engineering keywords before vectorization."""
    t_low = (title or "").lower()
    inferred_tags = []

    tag_map = {
        ("embedded", "firmware", "fpga", "rtos", "microcontroller", "iot", "hardware", "pcb", "verilog", "vhdl"):
            "Embedded Systems, Firmware, Hardware Engineering, Electronics, IoT, C/C++, Microcontroller, FPGA, RTL",
        ("electronic", "electronics", "telecommunication", "telecom", "rf", "instrumentation", "biomedical"):
            "Electronics Engineering, Telecommunications, Hardware, Embedded, Signal Processing, Instrumentation, PCB",
        ("electrical", "power", "hv", "lv", "mep", "substation", "grid", "solar", "renewable", "scada", "bess", "energy"):
            "Electrical Engineering, Power Systems, Solar & Renewable Energy, MEP, SCADA, Automation, Control Systems",
        ("mechanical", "mechatronics", "hvac", "manufacturing", "plant", "maintenance", "automotive", "robotics", "automation", "plc"):
            "Mechanical Engineering, Mechatronics, Industrial Automation, PLC/SCADA, Robotics, Manufacturing, CAD, Maintenance",
        ("civil", "structural", "qs", "quantity surveyor", "site", "construction", "draughtsman", "bim", "geotechnical"):
            "Civil Engineering, Structural Design, Construction Management, Site Engineering, Quantity Surveying, BIM",
        ("software", "full stack", "fullstack", "backend", "frontend", "react", "node", ".net", "java", "python", "golang", "mobile", "flutter", "ios", "android", "developer"):
            "Software Engineering, Full-Stack Development, Backend, Frontend, Web, Cloud, API, Microservices",
        ("ai", "ml", "machine learning", "data", "analytics", "bi", "scientist", "nlp", "vision", "deep learning", "llm", "genai"):
            "Data Science, Artificial Intelligence, Machine Learning, Computer Vision, Data Engineering, Analytics, Python, SQL",
        ("devops", "sre", "cloud", "aws", "azure", "kubernetes", "docker", "linux", "sysadmin", "network", "security", "cyber", "soc", "noc"):
            "DevOps, Cloud Infrastructure, Site Reliability, Cyber Security, Network Engineering, System Administration",
        ("qa", "quality assurance", "test", "sdet", "selenium", "cypress"):
            "Quality Assurance, Software Testing, Test Automation, SDET, QA Engineering"
    }

    for keywords, tags in tag_map.items():
        if any(k in t_low for k in keywords):
            inferred_tags.append(tags)

    domain_str = " | ".join(inferred_tags) if inferred_tags else "Engineering, Technology, IT, Technical Role"
    return f"Job Title: {title} | Company: {company} | Location: {location} | Domain Keywords: {domain_str} | Source: {source}"


def fetch_unvectorized_jobs() -> list[dict]:
    """Fetches remaining jobs lacking embeddings in 1,000-row chunks."""
    all_rows = []
    page_size = 1000
    offset = 0

    while True:
        res = (
            supabase.table("jobs")
            .select("ref_no, title, company, location, source")
            .is_("embedding", "null")
            .range(offset, offset + page_size - 1)
            .execute()
        )
        batch = res.data or []
        all_rows.extend(batch)

        if len(batch) < page_size:
            break
        offset += page_size

    return all_rows


def backfill_remaining():
    print("[*] Querying database for listings where embedding IS NULL...")
    pending_jobs = fetch_unvectorized_jobs()
    total = len(pending_jobs)

    if total == 0:
        print("[✔] All active jobs in your database are already 100% vectorized!")
        return

    # 45 items per batch stays safely under the 100 items/minute Free Tier cap
    batch_size = 45
    total_batches = (total + batch_size - 1) // batch_size
    print(f"[*] Found {total} unvectorized listings.")
    print(f"[*] Processing {total_batches} batches at {batch_size} items/batch...\n")

    for i in range(0, total, batch_size):
        batch_num = (i // batch_size) + 1
        chunk = pending_jobs[i:i + batch_size]

        texts = [
            build_enriched_embedding_text(
                title=j.get("title", ""),
                company=j.get("company", ""),
                location=j.get("location", ""),
                source=j.get("source", "")
            )
            for j in chunk
        ]

        print(f"[*] Vectorizing batch {batch_num}/{total_batches} ({len(chunk)} records)...")

        for attempt in range(5):
            try:
                resp = ai_client.models.embed_content(
                    model="gemini-embedding-001",
                    contents=texts,
                    config=types.EmbedContentConfig(output_dimensionality=768)
                )

                vectors = [emb.values for emb in resp.embeddings]

                for job, vec in zip(chunk, vectors):
                    supabase.table("jobs").update({"embedding": vec}).eq("ref_no", job["ref_no"]).execute()

                print(f"[+] Saved batch {batch_num}/{total_batches} successfully.")
                # Pace at 22s between 45-item batches (~90 items/min) to avoid 429 errors
                time.sleep(22.0)
                break

            except Exception as e:
                err_msg = str(e)

                # Stop immediately if project is denied access (403) or key is invalid
                if "PERMISSION_DENIED" in err_msg or "403" in err_msg or "API_KEY_INVALID" in err_msg:
                    print(f"\n[🛑] API Access Error (403/Key Issue): {err_msg}")
                    return

                # Only stop on TRUE daily quota exhaustion (PerDay metric)
                if "PerDay" in err_msg:
                    print("\n[🛑] True Daily limit (1,000/day) hit on this API key. Process halted.")
                    print("[*] Completed batches are saved. Switch keys or wait for reset to resume.")
                    return

                # Handle 1-minute rate limits
                if "RESOURCE_EXHAUSTED" in err_msg or "429" in err_msg:
                    match = re.search(r'Please retry in ([\d\.]+)s', err_msg)
                    wait_time = float(match.group(1)) + 4.0 if match else 45.0
                    print(f"[!] 1-minute rate limit reached. Pausing {wait_time:.1f}s (attempt {attempt + 1}/5)...")
                    time.sleep(wait_time)
                else:
                    print(f"[!] Error on attempt {attempt + 1}: {err_msg}")
                    time.sleep(3)
                    if attempt == 4:
                        print(f"[X] Skipping batch {batch_num} due to repeated errors.")

    print("\n[+] Backfill completed! Checking final vector coverage...")
    total_count = supabase.table("jobs").select("ref_no", count="exact").limit(1).execute().count or 0
    embedded_count = supabase.table("jobs").select("ref_no", count="exact").not_.is_("embedding", "null").limit(1).execute().count or 0
    pct = (embedded_count / total_count * 100) if total_count > 0 else 100.0
    print(f"[✔] Database Status: {embedded_count} / {total_count} listings vectorized ({pct:.1f}%).")


if __name__ == "__main__":
    backfill_remaining()