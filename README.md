# ⚡ JobGPT | AI Engineering Career Platform for Sri Lanka

An end-to-end, AI-powered career matching engine and ATS CV optimization studio purpose-built for Sri Lankan engineering graduates and tech professionals. 

JobGPT aggregates active technical vacancies across **TopJobs.lk**, **LinkedIn**, and **XpressJobs**, pairs them with 768-dimensional vector embeddings, and leverages Google Gemini to deliver semantic candidate-job matching and actionable resume feedback.

---

## 🚀 Key Features

* **Conversational AI Job Search:** Natural language query matching against 1,800+ live vacancies. Users can search by domain, tools, tech stack, or target firms with or without attaching a resume.
* **Hybrid Retrieval Architecture:** Combines high-speed cosine vector similarity search (`pgvector` + `ivfflat`) with cross-platform keyword matching for balanced coverage across major Sri Lankan job portals.
* **Direct Portal Deep-Linking:** Automated link resolution bypassing session timeouts, including zero-padded 10-digit TopJobs servlet endpoints (`JobAdvertismentServlet?jc=...`) with automated search fallbacks.
* **AI CV Optimizer & ATS Keyword Studio:** Evaluates uploaded PDF resumes against industry ATS benchmarks, identifies critical keyword gaps, and rewrites bullet points using the Google XYZ formula (*"Accomplished [X] as measured by [Y] by doing [Z]"*).
* **Automated Cloud Scraping (CI/CD):** Scheduled GitHub Actions runner (`daily_sync.yml`) scraping and vectorizing new job listings every morning at 05:30 AM SLST.
* **Tiered Access Engine:** Server-level daily usage enforcement (Free: 5 searches + 1 CV/day; Pro: 10 searches + 2 CVs/day) with persistent state tracking in Supabase and zero hardcoded secrets.

---

## 🛠️ Tech Stack

* **Frontend & UI:** [Streamlit](https://streamlit.io/) (Custom Dark Theme, responsive auto sidebar)
* **LLM & Embeddings:** [Google Gemini API](https://ai.google.dev/) (`gemini-3.5-flash-lite`, `gemini-embedding-001`)
* **Vector Database:** [Supabase](https://supabase.com/) (PostgreSQL with `pgvector`)
* **Web Scraping:** [Playwright](https://playwright.dev/python/) (Headless Chromium) & [BeautifulSoup4](https://www.crummy.com/software/BeautifulSoup/)
* **Document Parsing:** [PyPDF](https://pypdf.readthedocs.io/)
* **Validation & Schemas:** [Pydantic v2](https://docs.pydantic.dev/)

---