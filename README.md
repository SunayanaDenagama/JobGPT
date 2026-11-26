# ⚡ JobGPT | AI Engineering Career Platform for Sri Lanka

An end-to-end, AI-powered career matching engine and ATS CV optimization studio purpose-built for Sri Lankan engineering graduates and tech professionals. 

JobGPT aggregates active technical vacancies across **TopJobs.lk**, **LinkedIn**, and **XpressJobs**, pairs them with 768-dimensional vector embeddings, and leverages Google Gemini to deliver semantic candidate-job matching and actionable resume feedback.

---

## 🚀 Key Features

* **Conversational AI Job Search:** Natural language query matching against 1,800+ live vacancies. Users can search by domain, tools, tech stack, or target firms with or without attaching a resume.
* **Hybrid Retrieval Architecture:** Combines high-speed cosine vector similarity search (`pgvector` + `ivfflat`) with cross-platform keyword matching for balanced coverage across major Sri Lankan job portals.
* **Direct Portal Deep-Linking:** Zero-padded 10-digit TopJobs servlet endpoints (`JobAdvertismentServlet?jc=...`) with a separate reference-search link if the direct page doesn't load.
* **AI CV Optimizer & ATS Keyword Studio:** Evaluates uploaded PDF resumes against industry ATS benchmarks, identifies critical keyword gaps, and rewrites bullet points using the Google XYZ formula (*"Accomplished [X] as measured by [Y] by doing [Z]"*).
* **Automated Cloud Scraping (CI/CD):** Scheduled GitHub Actions runner (`daily_sync.yml`) scraping and vectorizing new job listings every morning at 05:30 AM SLST.
* **Tiered Access Engine:** Server-level daily usage enforcement (Free: 5 searches + 1 CV/day; Pro: 10 searches + 2 CVs/day) with persistent state tracking in Supabase and zero hardcoded secrets.

---

## 🛠️ Tech Stack

* **Frontend & UI:** [Streamlit](https://streamlit.io/) (Custom Dark Theme, responsive auto sidebar)
* **LLM & Embeddings:** [Google Gemini API](https://ai.google.dev/) (discovered Gemini Flash models for structured responses; `gemini-embedding-001` for 768-dimensional vectors)
* **Vector Database:** [Supabase](https://supabase.com/) (PostgreSQL with `pgvector`)
* **Web Scraping:** [Playwright](https://playwright.dev/python/) (Headless Chromium) & [BeautifulSoup4](https://www.crummy.com/software/BeautifulSoup/)
* **Document Parsing:** [PyPDF](https://pypdf.readthedocs.io/)
* **Validation & Schemas:** [Pydantic v2](https://docs.pydantic.dev/)

---

## Run locally

1. With Python 3.11 or newer, install dependencies: `python -m pip install -r requirements.txt`.
2. Set `SUPABASE_URL`, `SUPABASE_KEY`, and `GEMINI_API_KEY` in `.env` or `.streamlit/secrets.toml`. Use the existing JobGPT Supabase project with the jobs, quota, and analytics tables and matching RPC.
3. Run `python -m streamlit run app.py`.

Job Search uses keyword and vector retrieval in parallel by default, without waiting for text generation. **Add AI fit scores** optionally reranks a short list; if Gemini is slow or unavailable, the app shows the retrieved listings without fit scores. The landing page shows the count and newest listings from the regularly synced index. Confirm availability on the original portal.

The app discovers versioned Flash models that advertise `generateContent` for your key, prefers Flash-Lite when available, and can fall back from `v1` to `v1beta` within the generation time budget. Embeddings stay on `v1`. You can optionally set `GEMINI_MODEL` to prefer a specific discovered model on the available endpoint. To see what your key advertises, run `python list_models.py` with `GEMINI_API_KEY` in your environment or `.env`. CV Studio needs a working generation model.

Run the offline regression tests with `python -m unittest discover -s tests`.
