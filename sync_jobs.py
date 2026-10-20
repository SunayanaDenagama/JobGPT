import os
import re
import time
import httpx
from datetime import datetime, timezone, timedelta
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
# HELPER: Enrich Job Text Before Vectorization
# ==========================================
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


# ==========================================
# 1. TOPJOBS: Intelligent Pruning & Accurate Date Filtering
# ==========================================
def fetch_topjobs():
    categories = [
        ("SDQ", "Software/AI/Web/QA"),
        ("MAE", "Mechanical/Electrical/Power"),
        ("HNS", "Embedded/Networks/Hardware/Systems"),
        ("TEL", "Telecom/Electronics"),
        ("ENG", "Engineering/Operations/Energy"),
        ("CVA", "Civil/Structural/Architecture"),
        ("MFG", "Manufacturing/Industrial/Automation"),
        ("MED", "Biomedical/Healthcare Equipment")
    ]
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
    }

    # Non-engineering positions to filter out immediately
    EXCLUDED_TERMS = re.compile(
        r'\b(driver|cashier|security guard|clerk|cook|cleaner|waiter|receptionist|typist|peon|janitor|caretaker|steward|housekeeping|tailor|butcher|barista)\b',
        re.I
    )

    jobs = []
    seen = set()
    today_date = datetime.now().date()

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

                        # Skip non-technical / service vacancies
                        if EXCLUDED_TERMS.search(title):
                            continue

                        all_strings = list(position_col.stripped_strings)
                        company = all_strings[-1] if len(all_strings) > 1 else "Sri Lanka Enterprise"
                        town = tds[-1].get_text(strip=True) or "Sri Lanka"
                        closing_date = tds[-2].get_text(strip=True) if len(tds) >= 2 else ""

                        # Skip clearly expired listings (handles 'Fri Oct 16 2026', '16-Oct-2026', etc.)
                        if closing_date:
                            is_expired = False
                            for fmt in ("%a %b %d %Y", "%d-%b-%Y", "%b %d %Y"):
                                try:
                                    exp_dt = datetime.strptime(closing_date.strip(), fmt).date()
                                    if exp_dt < today_date:
                                        is_expired = True
                                    break
                                except ValueError:
                                    continue
                            if is_expired:
                                continue

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
        ("33", "Telecom & Electronics"),
        ("14", "Construction & Civil Engineering"),
        ("22", "Manufacturing & Operations")
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

                    for _ in range(5):
                        page.evaluate("window.scrollBy(0, 1200);")
                        page.wait_for_timeout(900)

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
        "Embedded Engineer",
        "Embedded Systems",
        "Firmware Engineer",
        "Hardware Engineer",
        "IoT Engineer",
        "FPGA",
        "Robotics Engineer",
        "Electronic Engineer",
        "Telecommunication Engineer",
        "Electrical Engineer",
        "Power Systems",
        "Solar Engineer",
        "SCADA",
        "Mechanical Engineer",
        "Mechatronics",
        "Automation Engineer",
        "Machine Learning",
        "AI Engineer",
        "Data Scientist",
        "Data Engineer",
        "Computer Vision",
        "DevOps Engineer",
        "Cloud Engineer",
        "Cyber Security Engineer",
        "Network Engineer",
        "QA Engineer",
        "Biomedical Engineer",
        "Civil Engineer",
        "Quantity Surveyor"
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
                time.sleep(0.55)
            except Exception as e:
                print(f"[!] LinkedIn error ({kw} @ {page_offset}): {e}")
                break

    print(f"[+] LinkedIn total collected across matrix: {len(jobs)}")
    return jobs


# ==========================================
# 4. SMART PURGE: REMOVE EXPIRED & CLOSED LISTINGS
# ==========================================
def purge_expired_and_closed_jobs(live_scraped_jobs: list[dict]):
    """
    Safely purges expired listings without removing genuine available jobs:
    1. Removes TopJobs listings where closing_date ('Fri Oct 16 2026' or 'DD-Mon-YYYY') < today.
    2. Removes TopJobs and XpressJobs in DB that were removed from the live website today.
    3. Removes remaining listings older than 35 days.
    """
    print("\n[*] Auditing database to purge expired and closed vacancies...")
    today_date = datetime.now(timezone.utc).date()

    # Step 1: Remove any records older than 35 days
    try:
        cutoff_35d = (datetime.now(timezone.utc) - timedelta(days=35)).isoformat()
        supabase.table("jobs").delete().lt("created_at", cutoff_35d).execute()
    except Exception as e:
        print(f"[!] 35-day cutoff note: {e}")

    # Step 2: Fetch existing records from Supabase in 1,000-row chunks
    db_jobs = []
    offset = 0
    while True:
        res = supabase.table("jobs").select("ref_no, closing_date, source, created_at").range(offset, offset + 999).execute()
        batch = res.data or []
        db_jobs.extend(batch)
        if len(batch) < 1000:
            break
        offset += 1000

    live_refs = {j["ref_no"] for j in live_scraped_jobs}
    has_live_tj = any(r.startswith("TJ-") for r in live_refs)
    has_live_xp = any(r.startswith("XP-") for r in live_refs)

    refs_to_delete = []

    for row in db_jobs:
        ref = row.get("ref_no", "")
        c_date = (row.get("closing_date") or "").strip()

        # Check A: Expiration date in TopJobs format (e.g. 'Fri Oct 16 2026', '16-Oct-2026') has passed
        is_expired = False
        for fmt in ("%a %b %d %Y", "%d-%b-%Y", "%b %d %Y"):
            try:
                exp_dt = datetime.strptime(c_date, fmt).date()
                if exp_dt < today_date:
                    is_expired = True
                break
            except ValueError:
                continue

        if is_expired:
            refs_to_delete.append(ref)
            continue

        # Check B: Live site reconciliation (TopJobs & XpressJobs closed / no longer on the board)
        if has_live_tj and ref.startswith("TJ-") and ref not in live_refs:
            refs_to_delete.append(ref)
            continue

        if has_live_xp and ref.startswith("XP-") and ref not in live_refs:
            refs_to_delete.append(ref)
            continue

    # Delete in batches of 100
    if refs_to_delete:
        print(f"[*] Removing {len(refs_to_delete)} expired/closed vacancies from database...")
        for i in range(0, len(refs_to_delete), 100):
            chunk = refs_to_delete[i:i + 100]
            try:
                supabase.table("jobs").delete().in_("ref_no", chunk).execute()
            except Exception as e:
                print(f"[!] Error deleting expired chunk: {e}")
        print(f"[+] Successfully purged {len(refs_to_delete)} closed vacancies!")
    else:
        print("[+] All database listings are active and available.")


# ==========================================
# 5. INCREMENTAL EMBEDDINGS & BATCH UPSERT
# ==========================================
def embed_new_jobs(new_jobs: list[dict]) -> list[dict]:
    """Generates 768-dim enriched embeddings for new listings with safe quota backoff."""
    if not new_jobs:
        return []

    print(f"[*] Generating enriched vector embeddings for {len(new_jobs)} new listings...")
    batch_size = 60
    embedded_jobs = []

    for i in range(0, len(new_jobs), batch_size):
        chunk = new_jobs[i:i + batch_size]
        texts = [
            build_enriched_embedding_text(
                title=j.get("title", ""),
                company=j.get("company", ""),
                location=j.get("location", ""),
                source=j.get("source", "")
            )
            for j in chunk
        ]

        success = False
        attempts = 0
        while not success and attempts < 4:
            attempts += 1
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
                time.sleep(1.2)
                success = True
            except Exception as e:
                err_msg = str(e)
                print(f"[!] Embedding API rate limit or quota warning: {err_msg}")
                if "PerDay" in err_msg:
                    print("[!] Daily embedding quota limit reached. Saving new jobs without embeddings for now.")
                    embedded_jobs.extend(chunk)
                    return embedded_jobs
                elif "RESOURCE_EXHAUSTED" in err_msg or "429" in err_msg:
                    print("[!] Rate limit reached. Sleeping 45s before retry...")
                    time.sleep(45)
                else:
                    print("[!] Non-quota error encountered. Saving batch without embeddings.")
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

    # Automatically purge closed vacancies before checking/inserting
    purge_expired_and_closed_jobs(all_scraped)

    print(f"\n[*] Total active scraped jobs: {len(all_scraped)}. Inspecting database for existing records...")

    scraped_refs = [j["ref_no"] for j in all_scraped]
    existing_refs = set()

    check_batch = 100
    for i in range(0, len(scraped_refs), check_batch):
        sub_refs = scraped_refs[i:i + check_batch]
        res = supabase.table("jobs").select("ref_no").in_("ref_no", sub_refs).execute()
        for r in (res.data or []):
            existing_refs.add(r["ref_no"])

    new_jobs = [j for j in all_scraped if j["ref_no"] not in existing_refs]
    existing_jobs = [j for j in all_scraped if j["ref_no"] in existing_refs]

    print(f"[*] Found {len(new_jobs)} brand new listings. {len(existing_jobs)} already exist in database.")

    ready_new_jobs = embed_new_jobs(new_jobs) if new_jobs else []
    combined = ready_new_jobs + existing_jobs

    print(f"[*] Upserting {len(combined)} active listings into Supabase...")
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