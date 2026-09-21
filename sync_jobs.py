import os
import re
import time
import httpx
from datetime import datetime
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
from supabase import create_client, Client
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not SUPABASE_URL or not SUPABASE_KEY or not GEMINI_API_KEY:
    raise ValueError("Missing Supabase or Gemini credentials in .env file!")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
ai_client = genai.Client(api_key=GEMINI_API_KEY)

# ==========================================
# 1. TOPJOBS: Intelligent Pruning & Direct Attachment Routing
# ==========================================
def fetch_topjobs():
    categories = [
        ("SDQ", "Software/AI/Web"),
        ("MAE", "Mechanical/Electrical/Power"),
        ("HNS", "Embedded/Networks/Hardware"),
        ("TEL", "Telecom/Electronics"),
        ("ENG", "Engineering/Operations/Energy"),
        ("CVA", "Civil/Structural/Architecture"),
        ("MED", "Biomedical/Healthcare Equipment")
    ]
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
    }

    # Non-engineering positions to filter out immediately
    EXCLUDED_TERMS = re.compile(
        r'\b(driver|cashier|security|clerk|cook|cleaner|waiter|receptionist|typist|peon|janitor|caretaker)\b', 
        re.I
    )

    jobs = []
    seen = set()

    for fa_code, cat_name in categories:
        url = f"http://www.topjobs.lk/applicant/vacancybyfunctionalarea.jsp?FA={fa_code}"
        print(f"[*] Scraping TopJobs ({cat_name})...")
        try:
            response = httpx.get(url, headers=headers, timeout=18.0)
            soup = BeautifulSoup(response.text, "html.parser")

            for tr in soup.find_all("tr"):
                tds = tr.find_all("td")
                if len(tds) >= 5:
                    ref_text = tds[1].get_text(strip=True) if len(tds) > 1 else ""
                    if re.match(r'^\d{6,8}$', ref_text) and ref_text not in seen:
                        position_col = tds[2]
                        
                        title_tag = position_col.find("h2")
                        if title_tag:
                            title = title_tag.get_text(strip=True)
                        else:
                            lines = [l.strip() for l in position_col.stripped_strings if l.strip()]
                            cleaned = [l for l in lines if not re.match(r'^(\d+|DEFZZZ)', l)]
                            title = cleaned[0] if cleaned else "Technical Position"

                        # Heuristic Filter: Skip non-technical/service vacancies
                        if EXCLUDED_TERMS.search(title):
                            continue

                        all_strings = list(position_col.stripped_strings)
                        company = all_strings[-1] if len(all_strings) > 1 else "Sri Lanka Enterprise"
                        town = tds[-1].get_text(strip=True) or "Sri Lanka"
                        closing_date = tds[-2].get_text(strip=True) if len(tds) >= 2 else ""

                        # Skip clearly expired listings if date format matches DD-Mon-YYYY
                        if closing_date:
                            try:
                                exp_dt = datetime.strptime(closing_date, "%d-%b-%Y")
                                if exp_dt.date() < datetime.now().date():
                                    continue
                            except Exception:
                                pass

                        seen.add(ref_text)
                        
                        # Direct attachment launcher bypassing session redirect
                        clean_numeric_id = re.sub(r'^[A-Za-z]+-', '', ref_text).strip()
                        direct_job_url = f"http://www.topjobs.lk/emp/job/empjobattachement.jsp?ac=DEFZZZ&id={clean_numeric_id}"

                        jobs.append({
                            "ref_no": f"TJ-{clean_numeric_id}",
                            "title": title,
                            "company": company,
                            "location": town,
                            "closing_date": closing_date or "Active",
                            "apply_url": direct_job_url,
                            "source": "TopJobs.lk"
                        })
        except Exception as e:
            print(f"[!] TopJobs error ({cat_name}): {e}")

    print(f"[+] TopJobs verified active positions: {len(jobs)}")
    return jobs

# ==========================================
# 2. XPRESSJOBS: Multi-Page Sector Scraping
# ==========================================
def fetch_xpressjobs():
    print("[*] Scraping XpressJobs (IT, Hardware, Engineering, Telecom)...")
    sectors = [
        ("30", "IT & Software"),
        ("29", "Engineering & Technical"),
        ("33", "Telecom & Electronics")
    ]
    jobs = []
    seen = set()

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True, channel="chrome")
            context = browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
            )
            page = context.new_page()

            for sec_id, sec_name in sectors:
                url = f"https://xpress.jobs/Jobs?Sectors={sec_id}"
                print(f"    Scanning {sec_name}...")
                
                try:
                    page.goto(url, wait_until="domcontentloaded", timeout=35000)
                    page.wait_for_timeout(3000)

                    for _ in range(3):
                        page.evaluate("window.scrollBy(0, 1000);")
                        page.wait_for_timeout(1000)

                    links_data = page.eval_on_selector_all(
                        "a",
                        """elements => elements.map(el => ({
                            href: el.href || '',
                            text: el.innerText ? el.innerText.trim() : ''
                        }))"""
                    )

                    sector_captured = 0
                    for item in links_data:
                        href = item.get("href", "")
                        text = item.get("text", "")

                        match = re.search(r'/view/(\d+)', href, re.I)
                        if match:
                            job_id = match.group(1)
                            if job_id not in seen and len(text) > 4 and text.upper() not in ["APPLY NOW", "VIEW JOB", "APPLY"]:
                                seen.add(job_id)
                                sector_captured += 1

                                raw_lines = [l.strip() for l in text.split("\n") if l.strip()]
                                title = raw_lines[0]
                                content_lines = [l for l in raw_lines[1:] if not re.search(r'(\d+\s+days?\s+left|full-time|part-time)', l, re.I)]
                                company = content_lines[0] if content_lines else "Sri Lanka Employer"
                                location = content_lines[1] if len(content_lines) > 1 else "Sri Lanka"

                                day_match = re.search(r'(\d+\s+days?\s+left)', text, re.I)
                                closing_date = day_match.group(1) if day_match else "Active"

                                jobs.append({
                                    "ref_no": f"XP-{job_id}",
                                    "title": title,
                                    "company": company,
                                    "location": location,
                                    "closing_date": closing_date,
                                    "apply_url": f"https://xpress.jobs/Jobs/View/{job_id}",
                                    "source": "XpressJobs"
                                })

                    print(f"    [+] {sec_name}: captured {sector_captured} roles.")
                except Exception as e:
                    print(f"    [!] Error loading {sec_name}: {e}")

            browser.close()
    except Exception as e:
        print(f"[!] XpressJobs browser error: {e}")

    print(f"[+] XpressJobs total collected: {len(jobs)}")
    return jobs

# ==========================================
# 3. LINKEDIN: Granular Multi-Query Pagination
# ==========================================
def fetch_linkedin_jobs():
    print("[*] Scraping LinkedIn Sri Lanka (Expanded keyword matrix)...")
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9"
    }

    search_queries = [
        "Software Engineer",
        "Full Stack Developer",
        "Electrical Engineer",
        "Electronic Engineer",
        "Embedded Systems",
        "Firmware",
        "Power Systems",
        "Solar Engineer",
        "SCADA",
        "Machine Learning",
        "AI Engineer",
        "Computer Vision",
        "Biomedical Engineer",
        "Civil Engineer",
        "Mechanical Engineer",
        "Mechatronics",
        "Automation Engineer",
        "FPGA"
    ]

    jobs = []
    seen = set()

    for kw in search_queries:
        for page_offset in [0, 25, 50]:
            encoded_kw = kw.replace(" ", "%20")
            url = f"https://www.linkedin.com/jobs-guest/jobs/api/seeMoreJobPostings/search?keywords={encoded_kw}&location=Sri%20Lanka&start={page_offset}"
            
            try:
                response = httpx.get(url, headers=headers, timeout=12.0)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, "html.parser")
                    cards = soup.find_all("li")

                    if not cards:
                        break

                    for card in cards:
                        title_node = card.find("h3", class_="base-search-card__title")
                        company_node = card.find("h4", class_="base-search-card__subtitle")
                        link_node = card.find("a", class_="base-card__full-link")
                        loc_node = card.find("span", class_="job-search-card__location")

                        if title_node and company_node and link_node:
                            raw_link = link_node.get("href", "")
                            id_match = re.search(r'view/(\d+)', raw_link) or re.search(r'-(\d+)', raw_link)
                            if not id_match:
                                continue
                                
                            ref_id = id_match.group(1)
                            if ref_id in seen:
                                continue
                            seen.add(ref_id)

                            title = title_node.get_text(strip=True)
                            company = company_node.get_text(strip=True)
                            location = loc_node.get_text(strip=True) if loc_node else "Sri Lanka"
                            direct_linkedin_url = f"https://www.linkedin.com/jobs/view/{ref_id}"

                            jobs.append({
                                "ref_no": f"LI-{ref_id}",
                                "title": title,
                                "company": company,
                                "location": location,
                                "closing_date": "Recently Added",
                                "apply_url": direct_linkedin_url,
                                "source": "LinkedIn"
                            })
                time.sleep(0.6)
            except Exception as e:
                print(f"[!] LinkedIn error ({kw} @ {page_offset}): {e}")
                break

    print(f"[+] LinkedIn total collected across matrix: {len(jobs)}")
    return jobs

# ==========================================
# 4. INCREMENTAL EMBEDDINGS & BATCH UPSERT
# ==========================================
def embed_new_jobs(new_jobs: list[dict]) -> list[dict]:
    """Generates 768-dim embeddings for listings using gemini-embedding-001 with quota backoff."""
    if not new_jobs:
        return []

    print(f"[*] Generating vector embeddings for {len(new_jobs)} new listings...")
    batch_size = 30
    embedded_jobs = []

    for i in range(0, len(new_jobs), batch_size):
        chunk = new_jobs[i:i + batch_size]
        texts = [f"{j.get('title', '')} at {j.get('company', '')} in {j.get('location', '')}" for j in chunk]

        success = False
        while not success:
            try:
                resp = ai_client.models.embed_content(
                    model="gemini-embedding-001",
                    contents=texts,
                    config=types.EmbedContentConfig(output_dimensionality=768)
                )
                vectors = [emb.values for emb in resp.embeddings]
                for job, vec in zip(chunk, vectors):
                    job["embedding"] = vec
                embedded_jobs.extend(chunk)
                time.sleep(2)
                success = True
            except Exception as e:
                err_msg = str(e)
                print(f"[!] Embedding API rate limit or quota warning: {err_msg}")
                if "RESOURCE_EXHAUSTED" in err_msg or "429" in err_msg:
                    print("[!] Sleeping 65s before retrying batch...")
                    time.sleep(65)
                else:
                    # Non-quota error, save without embedding to prevent blocking execution
                    print("[!] Bypassing embedding for this batch due to error.")
                    embedded_jobs.extend(chunk)
                    success = True

    return embedded_jobs

def sync_all():
    all_scraped = []
    all_scraped.extend(fetch_topjobs())
    all_scraped.extend(fetch_xpressjobs())
    all_scraped.extend(fetch_linkedin_jobs())

    if not all_scraped:
        print("[!] No listings found to sync.")
        return

    print(f"\n[*] Total scraped jobs: {len(all_scraped)}. Inspecting database for existing records...")

    # Identify which listings already exist in Supabase
    scraped_refs = [j["ref_no"] for j in all_scraped]
    existing_refs = set()
    
    # Supabase in_ query chunking to avoid query string length limits
    check_batch = 100
    for i in range(0, len(scraped_refs), check_batch):
        sub_refs = scraped_refs[i:i + check_batch]
        res = supabase.table("jobs").select("ref_no").in_("ref_no", sub_refs).execute()
        for r in (res.data or []):
            existing_refs.add(r["ref_no"])

    new_jobs = [j for j in all_scraped if j["ref_no"] not in existing_refs]
    existing_jobs = [j for j in all_scraped if j["ref_no"] in existing_refs]

    print(f"[*] Found {len(new_jobs)} brand new listings. {len(existing_jobs)} already exist in database.")

    # Vectorize strictly the newly discovered listings
    ready_new_jobs = embed_new_jobs(new_jobs) if new_jobs else []

    # Combine listings (new jobs have fresh embeddings; existing jobs preserve their fields)
    combined = ready_new_jobs + existing_jobs

    print(f"[*] Upserting {len(combined)} listings into Supabase...")
    batch_size = 100
    for i in range(0, len(combined), batch_size):
        chunk = combined[i:i + batch_size]
        try:
            supabase.table("jobs").upsert(chunk, on_conflict="ref_no").execute()
            print(f"    Upserted batch {i // batch_size + 1} ({len(chunk)} records)")
        except Exception as e:
            print(f"[!] Batch upsert error: {e}")

    print("\n[+] Daily scraping, vectorization, and database sync complete!")

if __name__ == "__main__":
    sync_all()