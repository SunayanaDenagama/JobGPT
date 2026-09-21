import httpx
from bs4 import BeautifulSoup
import re
import json

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}

# Software, Web, QA & IT Category URL
URL = "http://www.topjobs.lk/applicant/vacancybyfunctionalarea.jsp?FA=SDQ"

print(f"[*] Fetching live TopJobs vacancies...")
response = httpx.get(URL, headers=HEADERS, timeout=15.0)

soup = BeautifulSoup(response.text, "html.parser")
jobs = []

# Find all table rows
for tr in soup.find_all("tr"):
    # Every actual job row has an ID attribute (e.g. tr_0, tr_1...) or <td> with a Ref No.
    tds = tr.find_all("td")
    
    # Check if this row is an actual job entry (usually 5 to 7 columns)
    if len(tds) >= 5:
        # Col 0: # Number
        # Col 1: Job Ref No (all digits like '1547485')
        ref_text = tds[1].get_text(strip=True) if len(tds) > 1 else ""
        
        # Verify that Col 1 is indeed a numeric Job Ref No
        if re.match(r'^\d{6,8}$', ref_text):
            position_col = tds[2]
            
            # Position/Employer cell contains <h2> tag or text with Job Title and Company
            title_tag = position_col.find("h2")
            if title_tag:
                title = title_tag.get_text(strip=True)
            else:
                # Fallback: Split lines inside position cell
                lines = [line.strip() for line in position_col.stripped_strings if line.strip()]
                # Strip out internal numeric codes (like '0001549772 DEFZZZ')
                cleaned_lines = [l for l in lines if not re.match(r'^(\d+|DEFZZZ)', l)]
                title = cleaned_lines[0] if cleaned_lines else "Unknown Position"

            # Company name is typically right below the title in the same cell
            all_strings = list(position_col.stripped_strings)
            company = all_strings[-1] if len(all_strings) > 1 else "Unknown Company"
            
            # Extract town / location from the last column
            town = tds[-1].get_text(strip=True) or "Sri Lanka"
            
            # Extract opening and closing dates
            opening_date = tds[-3].get_text(strip=True) if len(tds) >= 3 else ""
            closing_date = tds[-2].get_text(strip=True) if len(tds) >= 2 else ""

            # TopJobs direct link using the reference ID
            job_url = f"http://www.topjobs.lk/emp/job/vacancy.jsp?id={ref_text}"

            jobs.append({
                "ref_no": ref_text,
                "title": title,
                "company": company,
                "location": town,
                "closing_date": closing_date,
                "apply_url": job_url,
                "source": "TopJobs.lk"
            })

print(f"\n[+] Successfully extracted {len(jobs)} real IT jobs from TopJobs!\n")

if jobs:
    print(json.dumps(jobs[:3], indent=2))