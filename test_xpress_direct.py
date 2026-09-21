import httpx
from bs4 import BeautifulSoup
import re
import json

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://xpress.jobs/",
}

# General vacancies URL
URL = "https://xpress.jobs/jobs"

print(f"[*] Connecting to XpressJobs: {URL}...")
try:
    response = httpx.get(URL, headers=HEADERS, timeout=20.0, follow_redirects=True)
    print(f"[*] Response Code: {response.status_code}")
    
    soup = BeautifulSoup(response.text, "html.parser")
    jobs = []
    seen = set()

    # Match all links containing /view/ or /apply/ with numeric ID
    links = soup.find_all("a", href=re.compile(r'/(view|apply)/\d+', re.I))
    print(f"[*] Raw matching links found: {len(links)}")

    for a in links:
        href = a.get("href", "")
        id_match = re.search(r'/(view|apply)/(\d+)', href, re.I)
        if not id_match:
            continue
            
        ref_id = id_match.group(2)
        if ref_id in seen:
            continue

        # Look for text inside the link or in parent elements
        title = a.get_text(strip=True)
        if not title or len(title) < 4 or title.lower() in ["apply now", "view job", "apply"]:
            # Search nearby container for title
            parent = a.find_parent(["div", "li", "article", "section"])
            if parent:
                heading = parent.find(["h2", "h3", "h4", "strong"])
                if heading:
                    title = heading.get_text(strip=True)

        if title and len(title) > 3 and title.lower() not in ["apply now", "view job", "apply"]:
            seen.add(ref_id)
            canonical_url = f"https://xpress.jobs/Jobs/View/{ref_id}"
            jobs.append({
                "ref_no": f"XP-{ref_id}",
                "title": title,
                "apply_url": canonical_url,
                "source": "XpressJobs"
            })

    print(f"[+] Successfully extracted {len(jobs)} jobs from XpressJobs!")
    if jobs:
        print("\n--- Sample Jobs ---")
        print(json.dumps(jobs[:3], indent=2))

except Exception as e:
    print(f"[!] Error: {e}")