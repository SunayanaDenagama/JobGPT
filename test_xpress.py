from playwright.sync_api import sync_playwright
import re
import json

def scrape_xpress_it_jobs():
    print("[*] Launching system Google Chrome via Playwright...")
    with sync_playwright() as p:
        # channel="chrome" connects directly to your installed desktop Chrome browser
        browser = p.chromium.launch(headless=True, channel="chrome")
        page = browser.new_page()

        # IT & Software Category (Sector 30)
        url = "https://xpress.jobs/Jobs?Sectors=30"
        print(f"[*] Navigating to {url}...")
        
        page.goto(url, wait_until="domcontentloaded", timeout=40000)
        # Give React 4 seconds to mount the vacancy components
        page.wait_for_timeout(4000)

        # Pull all anchor elements rendered on the page
        links_data = page.eval_on_selector_all(
            "a",
            """elements => elements.map(el => ({
                href: el.href || '',
                text: el.innerText ? el.innerText.trim() : ''
            }))"""
        )

        jobs = []
        seen = set()

        for item in links_data:
            href = item.get("href", "")
            text = item.get("text", "")

            # Look for /View/ or /view/ with a numeric job identifier
            match = re.search(r'/view/(\d+)', href, re.I)
            if match:
                job_id = match.group(1)
                
                # Filter out utility links like "APPLY NOW" or generic buttons
                if job_id not in seen and len(text) > 4 and text.upper() not in ["APPLY NOW", "VIEW JOB", "APPLY"]:
                    seen.add(job_id)
                    
                    # Split multiple lines (e.g. Title \n Company)
                    lines = [line.strip() for line in text.split("\n") if line.strip()]
                    title = lines[0]
                    company = lines[1] if len(lines) > 1 else "Company on XpressJobs"

                    jobs.append({
                        "ref_no": f"XP-{job_id}",
                        "title": title,
                        "company": company,
                        "location": "Sri Lanka",
                        "apply_url": f"https://xpress.jobs/Jobs/View/{job_id}",
                        "source": "XpressJobs"
                    })

        browser.close()

        print(f"\n[+] Successfully extracted {len(jobs)} live IT jobs from XpressJobs!")
        if jobs:
            print("\n--- Sample Extracted Vacancies ---")
            print(json.dumps(jobs[:3], indent=2))
            
        return jobs

if __name__ == "__main__":
    scrape_xpress_it_jobs()