import os
import re
import json
import time
import hashlib
from datetime import datetime, timezone
import streamlit as st
import streamlit.components.v1 as components
from pypdf import PdfReader
from pydantic import BaseModel, Field
from supabase import create_client, Client
from google import genai
from google.genai import types
from google.genai.errors import APIError
from dotenv import load_dotenv

load_dotenv()

# Read credentials
sb_url = os.getenv("SUPABASE_URL") or st.secrets.get("SUPABASE_URL")
sb_key = os.getenv("SUPABASE_KEY") or st.secrets.get("SUPABASE_KEY")
gemini_key = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY")

if not sb_url or not sb_key or not gemini_key:
    st.error("Missing credentials! Verify your .env file or Streamlit secrets.")
    st.stop()

supabase: Client = create_client(sb_url, sb_key)

# Configures the client to use the stable v1 API and direct API key authentication
ai_client = genai.Client(
    api_key=gemini_key.strip(),
    http_options=types.HttpOptions(api_version="v1")
)

# ----------------- TIER QUOTAS & USAGE ENGINE -----------------
TIER_LIMITS = {
    "free": {"chat": 5, "cv": 1},
    "pro":  {"chat": 10, "cv": 2}
}

def get_client_identifier() -> str:
    """Identifies unique client via IP hash or session fallback."""
    try:
        forwarded = st.context.headers.get("X-Forwarded-For")
        if forwarded:
            ip = forwarded.split(",")[0].strip()
        else:
            ip = st.context.headers.get("Remote-Addr", "unknown_user")
    except Exception:
        ip = "local_user"
    return hashlib.sha256(ip.encode()).hexdigest()[:16]

def get_current_usage():
    """Fetches or initializes the daily usage record for this client."""
    user_id = get_client_identifier()
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    record_id = f"{user_id}_{today}"
    is_pro = st.session_state.get("is_pro", False)

    try:
        res = supabase.table("user_usage").select("*").eq("id", record_id).execute()
        if res.data:
            return res.data[0]
        else:
            new_record = {
                "id": record_id,
                "usage_date": today,
                "chat_count": 0,
                "cv_count": 0,
                "is_pro": is_pro
            }
            supabase.table("user_usage").insert(new_record).execute()
            return new_record
    except Exception:
        # Fallback to local session storage if table is not yet set up
        if "local_usage" not in st.session_state:
            st.session_state.local_usage = {"chat_count": 0, "cv_count": 0, "is_pro": is_pro}
        return st.session_state.local_usage

def check_and_increment_quota(action_type: str) -> tuple[bool, str]:
    """Validates quota and increments count if allowable."""
    user_id = get_client_identifier()
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    record_id = f"{user_id}_{today}"
    is_pro = st.session_state.get("is_pro", False)
    tier = "pro" if is_pro else "free"
    limit = TIER_LIMITS[tier][action_type]

    usage = get_current_usage()
    current_count = usage.get(f"{action_type}_count", 0)

    if current_count >= limit:
        action_name = "Job Searches" if action_type == "chat" else "CV Optimizations"
        if not is_pro:
            return False, f"⚠️ Daily free limit reached ({limit} {action_name}). Unlock **JobGPT Pro** in the left sidebar or come back tomorrow!"
        else:
            return False, f"⚠️ Daily Pro limit reached ({limit} {action_name}) for today. Resets tomorrow at 00:00 UTC."

    new_count = current_count + 1
    try:
        supabase.table("user_usage").update({
            f"{action_type}_count": new_count,
            "is_pro": is_pro,
            "updated_at": datetime.now(timezone.utc).isoformat()
        }).eq("id", record_id).execute()
    except Exception:
        if "local_usage" in st.session_state:
            st.session_state.local_usage[f"{action_type}_count"] = new_count

    return True, ""

# ----------------- PYDANTIC SCHEMAS -----------------
class JobMatch(BaseModel):
    ref_no: str
    title: str
    company: str
    location: str
    apply_url: str
    match_score: int = Field(description="Relevance percentage between 0 and 100")
    domain_category: str = Field(description="Domain area, e.g. Civil, Mechanical, Embedded, Power, AI/ML, Software")
    fit_reason: str = Field(description="2 concise sentences detailing why this role fits the user query or CV")
    skill_gaps: str = Field(description="Key tools, frameworks, or domain requirements to prepare")

class MatchReport(BaseModel):
    matched_jobs: list[JobMatch]

class BulletImprovement(BaseModel):
    original_area: str = Field(description="The project or experience item reviewed")
    critique: str = Field(description="Weakness in metric, action, or impact")
    improved_version: str = Field(description="STAR / Google XYZ formula rewrite")

class CVOptimizationReport(BaseModel):
    overall_score: int = Field(description="ATS Readiness score 0-100")
    summary_critique: str = Field(description="Overall strategic critique")
    crucial_missing_keywords: list[str] = Field(description="Keywords that should be added")
    suggested_bullet_revisions: list[BulletImprovement] = Field(description="3 to 5 bullet rewrites")
    high_impact_skills_to_learn: list[str] = Field(description="Technologies to learn next")

# ----------------- PAGE CONFIG & BRANDING -----------------
st.set_page_config(
    page_title="JobGPT | LK Engineering Career AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="auto"
)

# Suppress external referer to prevent session bounces on external job portals
st.markdown('<meta name="referrer" content="no-referrer">', unsafe_allow_html=True)

# Custom Styling
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2.75rem !important;
        padding-bottom: 2rem !important;
    }

    @media (max-width: 768px) {
        .block-container {
            padding-top: 2rem !important;
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }
    }

    section[data-testid="stSidebar"] .block-container {
        padding-top: 1.25rem !important;
    }

    .brand-header {
        display: flex;
        align-items: center;
        gap: 14px;
        margin-bottom: 0.15rem;
    }

    .main-title {
        font-size: 2.7rem;
        font-weight: 800;
        background: linear-gradient(90deg, #ff4b4b, #ff8533);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.5px;
        line-height: 1.1;
    }

    .bot-icon {
        background: rgba(255, 75, 75, 0.12);
        border: 1px solid rgba(255, 75, 75, 0.35);
        border-radius: 12px;
        padding: 6px;
        display: flex;
        align-items: center;
        justify-content: center;
    }

    .sub-title {
        font-size: 1.18rem;
        font-weight: 600;
        color: #d1d5db;
        margin-bottom: 0.75rem;
    }

    .hero-banner {
        background: rgba(255, 75, 75, 0.07);
        border: 1px solid rgba(255, 75, 75, 0.22);
        border-radius: 10px;
        padding: 0.85rem 1.15rem;
        margin-bottom: 1.25rem;
    }

    .hero-banner p {
        font-size: 1.02rem;
        margin: 0;
        color: #f3f4f6;
        font-weight: 500;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="brand-header">
        <div class="bot-icon">
            <svg width="38" height="38" viewBox="0 0 24 24" fill="none" stroke="#ff4b4b" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M12 8V4H8"></path>
                <rect width="16" height="12" x="4" y="8" rx="2"></rect>
                <path d="M2 14h2"></path>
                <path d="M20 14h2"></path>
                <path d="M15 13v2"></path>
                <path d="M9 13v2"></path>
            </svg>
        </div>
        <div class="main-title">JobGPT</div>
    </div>
    <div class="sub-title">⚡ Sri Lanka Engineering & Tech Career Agent</div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="hero-banner">
        <p>🎯 <strong>Tell me which kind of job you are looking for — I will find the best fits for you!</strong><br>
        <span style="font-size: 0.88rem; color: #9ca3af; font-weight: normal;">
        AI-powered matching across 1,800+ live technical vacancies from TopJobs.lk, LinkedIn, and XpressJobs.
        </span></p>
    </div>
    """,
    unsafe_allow_html=True
)

# Candidate Resume Preset
SUNAYANA_CV_PRESET = (
    "Sunayana Denagama - BSc. Eng (Hons) in Electrical & Electronic Engineering, University of Peradeniya (CGPA 3.33). "
    "Expertise: Embedded Systems (ESP32, Microchip Studio, C/C++), RTL Design & FPGA synthesis (Verilog, Vivado, 3x3 Sobel core), "
    "Power Systems (Windforce PLC: 10MW utility solar plants, tracking systems, MBUS/SACU/SCADA, BESS battery storage, ML solar forecasting), "
    "AI/ML & Agentic Systems (PyTorch, OpenCV, ClearSight ADAS vision restoration, Mamba, LLM Autonomous Agents), "
    "Biomedical & Robotics (Pressure-controlled prosthetic robotic hand, KiCad PCB design, Proteus)."
)

# Manage state
if "is_pro" not in st.session_state:
    st.session_state.is_pro = False
if "active_cv_text" not in st.session_state:
    st.session_state.active_cv_text = ""
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = [
        {
            "role": "assistant",
            "content": "👋 **Welcome to JobGPT!**\n\nTell me which kind of job you are looking for, and I will find the best fits for you. You can search by engineering specialization, specific tools, or target companies — with or without attaching a CV."
        }
    ]

# ----------------- BADGE & LINK RENDERERS -----------------
def get_source_badge(ref_no: str, source_str: str = "") -> str:
    ref = ref_no.upper()
    src = source_str.lower()
    if ref.startswith("TJ-") or "topjobs" in src:
        return ":green[**TopJobs**]"
    elif ref.startswith("LI-") or "linkedin" in src:
        return ":blue[**LinkedIn**]"
    elif ref.startswith("XP-") or "xpress" in src:
        return ":violet[**XpressJobs**]"
    return ":gray[**Direct**]"

def clean_apply_url(ref_no: str, original_url: str) -> str:
    ref = ref_no.upper()
    if ref.startswith("TJ-") or "topjobs.lk" in original_url:
        numeric_id = re.sub(r'[^0-9]+', '', ref_no).strip()
        padded_jc = numeric_id.zfill(10)
        return f"http://www.topjobs.lk/employer/JobAdvertismentServlet?rid=1&ac=DEFZZZ&jc={padded_jc}&ec=DEFZZZ&pg=applicant/vacancybyfunctionalarea.jsp"
    return original_url

def render_apply_button(url: str, ref_no: str, label: str = "View & Apply ↗"):
    target_url = clean_apply_url(ref_no, url)
    ref = ref_no.upper()
    
    if ref.startswith("TJ-") or "topjobs.lk" in target_url:
        numeric_id = re.sub(r'[^0-9]+', '', ref_no).strip()
        html_code = f"""
        <button onclick="launchTopJobs()" style="
            display: block;
            width: 100%;
            cursor: pointer;
            border: none;
            background-color: #ff4b4b;
            color: white;
            padding: 0.5rem 1rem;
            border-radius: 0.5rem;
            font-weight: 600;
            font-size: 0.875rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.12);
        ">{label}</button>

        <script>
        function launchTopJobs() {{
            var win = window.open('{target_url}', '_blank');
            setTimeout(function() {{
                if (!win || win.closed) {{
                    window.open('http://www.topjobs.lk/applicant/vacancybyfunctionalarea.jsp?FA=ALL&sQut={numeric_id}', '_blank');
                }}
            }}, 800);
        }}
        </script>
        """
    else:
        html_code = f"""
        <a href="{target_url}" target="_blank" rel="noreferrer noopener" style="
            display: block;
            width: 100%;
            text-align: center;
            background-color: #ff4b4b;
            color: white;
            padding: 0.5rem 1rem;
            border-radius: 0.5rem;
            text-decoration: none;
            font-weight: 600;
            font-size: 0.875rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.12);
        ">{label}</a>
        """
    components.html(html_code, height=45)

# ----------------- RELEVANCE & HYBRID RETRIEVAL -----------------
def extract_query_keywords(prompt_text, cv_text):
    stopwords = {"i", "want", "need", "looking", "for", "a", "an", "the", "in", "and", "or", "to", "of", "with", "jobs", "role", "roles", "engineer", "engineering", "show", "me", "find", "best", "matches"}
    clean_prompt = re.sub(r'[^a-zA-Z0-9\s]', ' ', prompt_text.lower())
    prompt_tokens = [w for w in clean_prompt.split() if len(w) > 2 and w not in stopwords]
    
    cv_tokens = []
    if cv_text.strip():
        domain_anchors = ["embedded", "firmware", "solar", "power", "electrical", "electronic", "verilog", "fpga", "scada", "civil", "mechanical", "biomedical", "python", "c++", "pytorch", "vision", "bess", "mamba", "robotics", "kicad"]
        cv_lower = cv_text.lower()
        cv_tokens = [term for term in domain_anchors if term in cv_lower]
        
    combined = list(dict.fromkeys(prompt_tokens + cv_tokens))
    return " ".join(combined[:12])

def retrieve_hybrid_jobs(user_query: str, cv_text: str, top_k: int = 100) -> list[dict]:
    results = []
    seen_refs = set()

    # 1. Semantic Vector Similarity Search
    try:
        combined_text = user_query.strip()
        if cv_text.strip():
            combined_text += f" | {cv_text[:250]}"

        emb_resp = ai_client.models.embed_content(
            model="gemini-embedding-001",
            contents=combined_text,
            config=types.EmbedContentConfig(output_dimensionality=768)
        )
        query_vec = emb_resp.embeddings[0].values

        rpc_res = supabase.rpc("match_jobs_semantic", {
            "query_embedding": query_vec,
            "match_threshold": 0.15,
            "match_count": top_k
        }).execute()

        for j in (rpc_res.data or []):
            if j["ref_no"] not in seen_refs:
                seen_refs.add(j["ref_no"])
                j["apply_url"] = clean_apply_url(j["ref_no"], j.get("apply_url", ""))
                results.append(j)
    except Exception as e:
        print(f"[*] Vector search skipped: {e}")

    # 2. Balanced Cross-Platform Supplement
    keywords = extract_query_keywords(user_query, cv_text)
    tj_count = sum(1 for j in results if j.get("ref_no", "").startswith("TJ-"))
    xp_count = sum(1 for j in results if j.get("ref_no", "").startswith("XP-"))

    for prefix, src_label in [("TJ-%", "TopJobs.lk"), ("XP-%", "XpressJobs")]:
        current_count = tj_count if prefix == "TJ-%" else xp_count
        if current_count < 15:
            try:
                query = supabase.table("jobs").select("ref_no, title, company, location, apply_url, source").ilike("ref_no", prefix)
                if keywords.split():
                    query = query.ilike("title", f"%{keywords.split()[0]}%")
                extra_res = query.order("created_at", desc=True).limit(20).execute()
                for j in (extra_res.data or []):
                    if j["ref_no"] not in seen_refs:
                        seen_refs.add(j["ref_no"])
                        j["apply_url"] = clean_apply_url(j["ref_no"], j.get("apply_url", ""))
                        results.append(j)
            except Exception as e:
                print(f"[*] Platform fill note ({src_label}): {e}")

    # 3. Final safety fill
    if len(results) < 30:
        try:
            fill_res = supabase.table("jobs").select("ref_no, title, company, location, apply_url, source").order("created_at", desc=True).limit(top_k - len(results)).execute()
            for j in (fill_res.data or []):
                if j["ref_no"] not in seen_refs:
                    seen_refs.add(j["ref_no"])
                    j["apply_url"] = clean_apply_url(j["ref_no"], j.get("apply_url", ""))
                    results.append(j)
        except Exception as e:
            print(f"[*] Safety fill note: {e}")

    return results

# ----------------- MULTI-MODEL FALLBACK CALLER -----------------
def call_gemini(prompt, schema):
    models = [
        "gemini-3.5-flash-lite",
        "gemini-2.5-flash",
        "gemini-flash-latest"
    ]
    last_err = ""
    for model_name in models:
        try:
            resp = ai_client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=schema,
                    temperature=0.1,
                ),
            )
            return schema.model_validate_json(resp.text), None
        except APIError as e:
            last_err = e.message
            time.sleep(1)
            continue
        except Exception as e:
            last_err = str(e)
            time.sleep(1)
            continue
    return None, last_err

# ----------------- SIDEBAR: NAVIGATION, PRO TIER & CV BUFFER -----------------
st.sidebar.markdown("### 🧭 Navigation")
nav_selection = st.sidebar.radio(
    "Select Feature:",
    ["💬 Job Match Chatbot", "📄 AI CV Optimizer & ATS Keywords"],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")

# PRO TIER / USAGE STATUS SECTION
st.sidebar.markdown("### ⚡ Subscription & Daily Quota")
current_usage = get_current_usage()
is_user_pro = st.session_state.is_pro
tier_name = "pro" if is_user_pro else "free"
max_chats = TIER_LIMITS[tier_name]["chat"]
max_cvs = TIER_LIMITS[tier_name]["cv"]
used_chats = current_usage.get("chat_count", 0)
used_cvs = current_usage.get("cv_count", 0)

if is_user_pro:
    st.sidebar.success(f"⭐ **JobGPT Pro Active**\n\n• Searches: **{used_chats} / {max_chats}** used\n• CV Reviews: **{used_cvs} / {max_cvs}** used")
    if st.sidebar.button("Log out of Pro", use_container_width=True):
        st.session_state.is_pro = False
        st.rerun()
else:
    st.sidebar.info(f"👤 **Free Plan**\n\n• Searches: **{used_chats} / {max_chats}** used\n• CV Reviews: **{used_cvs} / {max_cvs}** used")
    with st.sidebar.expander("🔑 Unlock Pro Access"):
        p_name = st.text_input("Username:", key="pro_name_in")
        p_pass = st.text_input("Password:", type="password", key="pro_pwd_in")
        if st.button("Activate Pro", use_container_width=True):
            # Read securely from environment / secrets
            valid_user = os.getenv("PRO_USER") or st.secrets.get("PRO_USER", "Mate80pro")
            valid_pass = os.getenv("PRO_PASSWORD") or st.secrets.get("PRO_PASSWORD", "2018")
            
            if p_name.strip() == valid_user and p_pass.strip() == valid_pass:
                st.session_state.is_pro = True
                st.success("Pro Activated! 10 searches + 2 CV reviews unlocked.")
                st.rerun()
            else:
                st.error("Incorrect credentials.")

st.sidebar.markdown("---")
st.sidebar.header("📋 Candidate CV (Optional)")
st.sidebar.caption("Leave blank to search purely by chat, or attach a CV to customize matching.")

col_b1, col_b2 = st.sidebar.columns(2)
with col_b1:
    if st.button("⚡ Load My CV", use_container_width=True):
        st.session_state.active_cv_text = SUNAYANA_CV_PRESET
        st.session_state.chat_messages.append(
            {"role": "assistant", "content": "Loaded Sunayana's CV into background context. Type your query or target criteria anytime!"}
        )
        st.rerun()

with col_b2:
    if st.button("🧹 Clear CV", use_container_width=True):
        st.session_state.active_cv_text = ""
        st.rerun()

uploaded_cv = st.sidebar.file_uploader("Attach Resume PDF (Optional):", type=["pdf"])
if uploaded_cv is not None:
    try:
        reader = PdfReader(uploaded_cv)
        extracted = "".join([p.extract_text() or "" for p in reader.pages])
        if extracted and extracted != st.session_state.active_cv_text:
            st.session_state.active_cv_text = extracted
            st.sidebar.success("CV loaded as supplementary context!")
            st.rerun()
    except Exception as e:
        st.sidebar.error(f"Error reading PDF: {e}")

st.session_state.active_cv_text = st.sidebar.text_area(
    "CV Context Buffer:",
    value=st.session_state.active_cv_text,
    placeholder="No CV loaded. Conversational search will use your direct chat queries.",
    height=140
)

# =================================================
# VIEW 1: CONVERSATIONAL JOB SEARCH CHATBOT
# =================================================
if nav_selection == "💬 Job Match Chatbot":
    # Quick-Filter Chips
    st.markdown("**Popular searches:**")
    q_col1, q_col2, q_col3, q_col4 = st.columns(4)
    with q_col1:
        if st.button("☀️ Solar & Renewable Power", use_container_width=True):
            st.session_state.quick_prompt = "Solar Power, BESS, and Renewable Energy Engineer"
    with q_col2:
        if st.button("📟 Embedded Systems & C++", use_container_width=True):
            st.session_state.quick_prompt = "Embedded Systems, Firmware, and C++ Microcontroller roles"
    with q_col3:
        if st.button("🏗️ Civil Site & Structural", use_container_width=True):
            st.session_state.quick_prompt = "Civil Site Engineer and Structural Design positions"
    with q_col4:
        if st.button("🤖 AI / ML & Python", use_container_width=True):
            st.session_state.quick_prompt = "Machine Learning, Computer Vision, and Python Developer"

    # Match setting sliders
    c_set1, c_set2 = st.columns([1, 1])
    with c_set1:
        match_count = st.slider("Target Matches to Display:", min_value=5, max_value=30, value=20, step=5)
    with c_set2:
        min_threshold = st.slider("Minimum Match Threshold (%):", min_value=30, max_value=85, value=50, step=5)

    if st.session_state.active_cv_text.strip():
        st.info("ℹ️ CV Context Active: JobGPT will match against both your prompt and CV background.", icon="📄")

    # Render Chat History
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "job_cards" in msg:
                for job in msg["job_cards"]:
                    with st.container(border=True):
                        c1, c2 = st.columns([3, 1])
                        badge = get_source_badge(job['ref_no'])
                        with c1:
                            st.subheader(f"{job['title']}")
                            st.markdown(f"🏢 **{job['company']}** | 📍 {job['location']} | 🌐 Source: {badge} | Ref: `{job['ref_no']}`")
                            st.markdown(f"**Domain:** `{job['domain']}`")
                            st.markdown(f"**Why it matches:** {job['fit']}")
                            if job.get("gaps"):
                                st.info(f"**Key skills to verify:** {job['gaps']}", icon="💡")
                        with c2:
                            st.metric("Fit Score", f"{job['score']}%")
                            render_apply_button(job['url'], job['ref_no'])

    # Determine Active Prompt (Input box or Quick Chip)
    active_prompt = None
    if "quick_prompt" in st.session_state and st.session_state.quick_prompt:
        active_prompt = st.session_state.pop("quick_prompt")

    chat_input_val = st.chat_input("Tell me which kind of job you are looking for...")
    user_prompt = active_prompt or chat_input_val

    if user_prompt:
        # Check quota prior to execution
        allowed, quota_msg = check_and_increment_quota("chat")
        if not allowed:
            st.warning(quota_msg, icon="🛑")
        else:
            st.session_state.chat_messages.append({"role": "user", "content": user_prompt})
            with st.chat_message("user"):
                st.markdown(user_prompt)

            with st.chat_message("assistant"):
                with st.spinner("Stage 1: Retrieving best matches via hybrid semantic + keyword search..."):
                    candidate_jobs = retrieve_hybrid_jobs(user_prompt, st.session_state.active_cv_text, top_k=100)

                with st.spinner(f"Stage 2: Gemini analyzing candidates to rank top {match_count}..."):
                    conversation_context = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.chat_messages])
                    cv_section = f"Candidate CV Background:\n{st.session_state.active_cv_text}" if st.session_state.active_cv_text.strip() else "Candidate CV: [None provided. Rely strictly on the chat query!]"

                    eval_prompt = f"""
                    You are JobGPT, an expert technical engineering recruiter in Sri Lanka.
                    
                    {cv_section}
                    
                    Conversation Context:
                    {conversation_context}
                    
                    LATEST USER SEARCH QUERY: "{user_prompt}"
                    
                    Candidate Vacancies Retrieved from Sri Lanka Job Database:
                    {json.dumps(candidate_jobs, indent=2)}
                    
                    Instructions:
                    1. Thoroughly evaluate these candidate vacancies against the user's latest search query and any CV context provided.
                    2. Select the top {match_count} best matching opportunities with a match score >= {min_threshold}%.
                    3. Sort the returned matches strictly from highest match score to lowest.
                    4. For each matched position, provide the match score, domain category, a clear 2-sentence explanation of why it fits, and any skill gaps to prepare.
                    """

                    report, err = call_gemini(eval_prompt, MatchReport)

                    if report and report.matched_jobs:
                        reply_text = f"Retrieved top candidates via **Hybrid Search** and selected the **top {len(report.matched_jobs)} matches** for your query:"
                        st.markdown(reply_text)

                        serialized_jobs = []
                        for item in sorted(report.matched_jobs, key=lambda x: x.match_score, reverse=True):
                            final_url = clean_apply_url(item.ref_no, item.apply_url)

                            job_data = {
                                "ref_no": item.ref_no,
                                "title": item.title,
                                "company": item.company,
                                "location": item.location,
                                "url": final_url,
                                "score": item.match_score,
                                "domain": item.domain_category,
                                "fit": item.fit_reason,
                                "gaps": item.skill_gaps
                            }
                            serialized_jobs.append(job_data)

                            badge = get_source_badge(item.ref_no)
                            with st.container(border=True):
                                c1, c2 = st.columns([3, 1])
                                with c1:
                                    st.subheader(f"{item.title}")
                                    st.markdown(f"🏢 **{item.company}** | 📍 {item.location} | 🌐 Source: {badge} | Ref: `{item.ref_no}`")
                                    st.markdown(f"**Domain:** `{item.domain_category}`")
                                    st.markdown(f"**Why it matches:** {item.fit_reason}")
                                    if item.skill_gaps:
                                        st.info(f"**Key skills to verify:** {item.skill_gaps}", icon="💡")
                                with c2:
                                    st.metric("Fit Score", f"{item.match_score}%")
                                    render_apply_button(final_url, item.ref_no)

                        st.session_state.chat_messages.append({
                            "role": "assistant",
                            "content": reply_text,
                            "job_cards": serialized_jobs
                        })
                    else:
                        err_msg = f"Could not complete matching: {err}" if err else f"No vacancies met the {min_threshold}% threshold for that query."
                        st.error(err_msg)
                        st.session_state.chat_messages.append({"role": "assistant", "content": err_msg})

# =================================================
# VIEW 2: AI CV OPTIMIZER
# =================================================
elif nav_selection == "📄 AI CV Optimizer & ATS Keywords":
    st.subheader("AI CV Optimizer & ATS Keyword Studio")
    st.markdown("Audits resume bullet points, scoring them against industry ATS requirements and the Google XYZ framework.")

    target_focus = st.text_input(
        "Target Role / Field for Optimization:",
        value="Embedded Systems & Renewable Power Engineer"
    )

    if st.button("Optimize My CV & Analyze Keywords 🔍", type="primary", use_container_width=True):
        if not st.session_state.active_cv_text.strip():
            st.warning("Please upload a PDF or click 'Load My CV' in the sidebar first to use the CV Optimizer.")
        else:
            allowed, quota_msg = check_and_increment_quota("cv")
            if not allowed:
                st.warning(quota_msg, icon="🛑")
            else:
                with st.spinner("Auditing resume against industry recruitment standards..."):
                    opt_prompt = f"""
                    You are JobGPT's principal technical hiring manager and engineering resume strategist.
                    
                    Target Specialization: {target_focus}
                    
                    Resume to Review:
                    {st.session_state.active_cv_text}
                    
                    Task:
                    1. Provide an ATS readiness score (0-100).
                    2. Summarize strategic strengths and critiques.
                    3. List 8 to 12 crucial ATS keywords to add for this engineering branch.
                    4. Select 3 to 5 actual bullet points from the CV and rewrite them using the Google XYZ formula: 'Accomplished [X] as measured by [Y] by doing [Z]'.
                    5. Highlight modern tools and standards to learn.
                    """

                    opt_report, opt_err = call_gemini(opt_prompt, CVOptimizationReport)

                    if opt_report:
                        st.metric("Resume ATS Industry Readiness", f"{opt_report.overall_score} / 100")
                        st.markdown(f"### 📋 Strategic Assessment\n{opt_report.summary_critique}")

                        st.markdown("---")
                        st.markdown("### 🔑 High-Impact ATS Keywords to Include")
                        kw_cols = st.columns(3)
                        for idx, kw in enumerate(opt_report.crucial_missing_keywords):
                            kw_cols[idx % 3].markdown(f"✔️ `{kw}`")

                        st.markdown("---")
                        st.markdown("### ✍️ Measurable Bullet Point Revisions (Google XYZ Formula)")
                        for rev in opt_report.suggested_bullet_revisions:
                            with st.expander(f"📌 Area: {rev.original_area}", expanded=True):
                                st.markdown(f"**Critique:** {rev.critique}")
                                st.success(f"**Recommended Rewrite:**\n\n{rev.improved_version}")

                        st.markdown("---")
                        st.markdown("### 🚀 Emerging Skills & Standards to Prioritize")
                        for tool in opt_report.high_impact_skills_to_learn:
                            st.markdown(f"• **{tool}**")
                    else:
                        st.error(f"Could not complete CV optimization: {opt_err}")