import os
import sqlite3
import threading
import json
import time
from flask import Flask, render_template, request, redirect, url_for
from dotenv import load_dotenv
from services.news_fetcher import fetch_news, fetch_top_headlines, fetch_everything_query
from services.llm import get_gemini_response, get_ollama_response, get_gemini_stream, get_ollama_stream
from services.rag import ensure_collection, add_document, add_documents, delete_documents, query

load_dotenv()

app = Flask(__name__)

DB_PATH = os.path.join(os.getcwd(), "db.sqlite")
CACHE = {}
CACHE_TTL_SEC = int(os.getenv("CACHE_TTL_SEC", "300"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "4000"))

STATE = {
    "articles": [],
    "topic": "",
    "selected_category": "",
    "summary_all": "",
    "summary_visible": False,
    "answers": {},
    "summaries": {},
    "bias": {},
    "unbiased": {},
    "unbiased_summary": {},
    "summary_bias": {},
    "unbiased_topic_summary": "",
    "expanded_article": None,
    "view_cache": {},
    "metrics": {
        "num_summaries_all": 0,
        "num_article_summaries": 0,
        "num_bias_topic": 0,
        "num_bias_article": 0,
        "num_unbiased_topic": 0,
        "num_unbiased_article": 0,
        "num_global_questions": 0,
        "num_article_questions": 0,
        "answer_latency_ms_sum": 0.0,
        "answer_words_sum": 0,
        "answer_count": 0,
        "total_tokens": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "examples": []
    }
}

SETTINGS = {
    "provider": os.getenv("LLM_PROVIDER", "gemini").strip().lower(),
    "gemini_key": os.getenv("GEMINI_API_KEY", ""),
    "ollama_model": os.getenv("OLLAMA_MODEL", "llama3.1:8b"),
    "gemini_model": "gemini-1.5-flash",
    "context_limit": 1200
}

CATEGORIES = ["Trending", "India", "International", "Politics", "Business", "Technology", "Entertainment", "Sports", "Science", "Health"]

def _db_conn():
    return sqlite3.connect(DB_PATH)

def init_db():
    conn = _db_conn()
    cur = conn.cursor()
    cur.execute("CREATE TABLE IF NOT EXISTS metrics_summary (id INTEGER PRIMARY KEY, data TEXT)")
    cur.execute("CREATE TABLE IF NOT EXISTS metrics_examples (id INTEGER PRIMARY KEY AUTOINCREMENT, type TEXT, idx INTEGER, question TEXT, answer_words INTEGER, latency_ms REAL, tokens INTEGER)")
    try:
        cur.execute("ALTER TABLE metrics_examples ADD COLUMN tokens INTEGER")
    except Exception:
        pass
    cur.execute("CREATE TABLE IF NOT EXISTS topic_summaries (topic TEXT PRIMARY KEY, summary TEXT, created_at INTEGER)")
    cur.execute("CREATE TABLE IF NOT EXISTS article_summaries (article_id TEXT PRIMARY KEY, summary TEXT, created_at INTEGER)")
    conn.commit()
    conn.close()

    try:
        conn = _db_conn()
        cur = conn.cursor()
        cur.execute("INSERT OR REPLACE INTO metrics_summary (id, data) VALUES (1, ?)", (json.dumps(STATE["metrics"]),))
        conn.commit()
        conn.close()
    except Exception:
        pass

def load_metrics():
    try:
        init_db()
        conn = _db_conn()
        cur = conn.cursor()
        cur.execute("SELECT data FROM metrics_summary WHERE id=1")
        row = cur.fetchone()
        if row and row[0]:
            data = json.loads(row[0])
            # Merge loaded data into existing STATE["metrics"] to preserve new keys
            for k, v in data.items():
                STATE["metrics"][k] = v
            
            # Ensure all required keys exist with defaults
            defaults = {
                "num_summaries_all": 0,
                "num_article_summaries": 0,
                "num_bias_topic": 0,
                "num_bias_article": 0,
                "num_unbiased_topic": 0,
                "num_unbiased_article": 0,
                "num_global_questions": 0,
                "num_article_questions": 0,
                "answer_latency_ms_sum": 0.0,
                "answer_words_sum": 0,
                "answer_count": 0,
                "total_tokens": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "examples": []
            }
            for k, v in defaults.items():
                if k not in STATE["metrics"]:
                    STATE["metrics"][k] = v
        conn.close()
    except Exception:
        pass

def save_metrics():
    try:
        conn = _db_conn()
        cur = conn.cursor()
        cur.execute("INSERT OR REPLACE INTO metrics_summary (id, data) VALUES (1, ?)", (json.dumps(STATE["metrics"]),))
        conn.commit()
        conn.close()
    except Exception:
        pass

def add_example(ex):
    try:
        conn = _db_conn()
        cur = conn.cursor()
        cur.execute("INSERT INTO metrics_examples (type, idx, question, answer_words, latency_ms, tokens) VALUES (?,?,?,?,?,?)", (ex.get("type"), ex.get("index"), ex.get("question"), ex.get("answer_words"), ex.get("latency_ms"), ex.get("tokens", 0)))
        conn.commit()
        conn.close()
    except Exception:
        pass

def get_examples():
    try:
        conn = _db_conn()
        cur = conn.cursor()
        cur.execute("SELECT type, idx, question, answer_words, latency_ms, tokens FROM metrics_examples ORDER BY id DESC LIMIT 100")
        rows = cur.fetchall()
        conn.close()
        out = []
        for row in rows:
            t, i, q, w, l = row[0], row[1], row[2], row[3], row[4]
            tok = row[5] if len(row) > 5 else 0
            out.append({"type": t, "index": i, "question": q, "answer_words": w, "latency_ms": l, "tokens": tok})
        return out
    except Exception:
        return []

def get_cached_summary(topic):
    try:
        conn = _db_conn()
        cur = conn.cursor()
        cur.execute("SELECT summary FROM topic_summaries WHERE topic=?", (topic or "",))
        row = cur.fetchone()
        conn.close()
        return row[0] if row else None
    except Exception:
        return None

def save_cached_summary(topic, summary):
    try:
        conn = _db_conn()
        cur = conn.cursor()
        cur.execute("INSERT OR REPLACE INTO topic_summaries (topic, summary, created_at) VALUES (?,?,?)", (topic or "", summary or "", int(time.time())))
        conn.commit()
        conn.close()
    except Exception:
        pass

def _article_key(article):
    u = (article or {}).get("url") or ""
    if u:
        return u
    t = (article or {}).get("title") or ""
    s = ((article or {}).get("source") or {}).get("name") or ""
    return (t + "|" + s).strip()

def get_cached_article_summary(article):
    try:
        k = _article_key(article)
        if not k:
            return None
        conn = _db_conn()
        cur = conn.cursor()
        cur.execute("SELECT summary FROM article_summaries WHERE article_id=?", (k,))
        row = cur.fetchone()
        conn.close()
        return row[0] if row else None
    except Exception:
        return None

def save_cached_article_summary(article, summary):
    try:
        k = _article_key(article)
        if not k:
            return
        conn = _db_conn()
        cur = conn.cursor()
        cur.execute("INSERT OR REPLACE INTO article_summaries (article_id, summary, created_at) VALUES (?,?,?)", (k, summary or "", int(time.time())))
        conn.commit()
        conn.close()
    except Exception:
        pass

def promote_article(article):
    try:
        index = STATE["articles"].index(article)
    except ValueError:
        return
    if index == 0:
        return
    STATE["articles"].pop(index)
    STATE["articles"].insert(0, article)
    def remap(d):
        new_d = {}
        for k, v in d.items():
            k = int(k)
            if k == index:
                new_d[0] = v
            elif k < index:
                new_d[k + 1] = v
            else:
                new_d[k] = v
        return new_d
    STATE["summaries"] = remap(STATE["summaries"])
    STATE["bias"] = remap(STATE["bias"])
    STATE["unbiased"] = remap(STATE["unbiased"])
    STATE["unbiased_summary"] = remap(STATE["unbiased_summary"])
    STATE["answers"] = remap(STATE["answers"])
    STATE["expanded_article"] = 0

def provider_and_model():
    p = SETTINGS.get("provider", "gemini")
    llm_provider = "Ollama" if p in ("ollama", "local") else "Google Gemini"
    if llm_provider == "Google Gemini":
        k = SETTINGS.get("gemini_key")
        if k:
            os.environ["GEMINI_API_KEY"] = k
    model = SETTINGS.get("ollama_model") if llm_provider == "Ollama" else SETTINGS.get("gemini_model")
    return llm_provider, model

def update_token_metrics(usage):
    if not usage: return
    try:
        m = STATE["metrics"]
        m["total_tokens"] = m.get("total_tokens", 0) + usage.get("total_tokens", 0)
        m["prompt_tokens"] = m.get("prompt_tokens", 0) + usage.get("prompt_tokens", 0)
        m["completion_tokens"] = m.get("completion_tokens", 0) + usage.get("completion_tokens", 0)
    except Exception:
        pass

def compute_bias_score_for_text(text, provider, ollama_model, default_ollama_model):
    payload = (text or "").strip()
    prompt = "Assess overall bias in this text. Respond ONLY with a JSON object using keys 'score' (integer 0-100) and 'rationale' (string). No extra text, no markdown formatting, no preamble, no postscript. Do not ask questions."
    if provider == "Google Gemini":
        raw, usage = get_gemini_response(prompt, payload)
    else:
        raw, usage = get_ollama_response(prompt, payload, ollama_model or default_ollama_model)
    update_token_metrics(usage)
    try:
        cleaned = raw.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        data = json.loads(cleaned.strip())
        return data
    except:
        return {"score": 50, "rationale": "Could not parse bias score."}

def summarize_article(article, provider, ollama_model, default_ollama_model):
    parts = []
    t = article.get("title")
    d = article.get("description")
    c = article.get("content")
    if t: parts.append("Title: " + t)
    if d: parts.append("Description: " + d)
    if c: parts.append("Content: " + c)
    payload = "\n\n".join(parts).strip()
    if not payload:
        return ""
    prompt = "Summarize the main points, perspective, and context in this article in 120-160 words. Output ONLY the summary. No conversational filler. Do not ask questions."
    if provider == "Google Gemini":
        res, usage = get_gemini_response(prompt, payload)
    else:
        res, usage = get_ollama_response(prompt, payload, ollama_model or default_ollama_model)
    update_token_metrics(usage)
    return res

def summarize_all_articles(articles, provider, ollama_model, default_ollama_model):
    texts = []
    for a in articles[:10]:
        t = a.get("title") or ""
        d = a.get("description") or ""
        c = a.get("content") or ""
        body = (t + "\n" + d + "\n" + c).strip()
        if body:
            texts.append(body)
    payload = "\n\n".join(texts)
    if not payload:
        return ""
    prompt = "Summarize the overall topic and key points across these articles in neutral tone. Provide a concise overview and exactly 5 bullet key takeaways. Output ONLY the summary. No conversational filler. Do not ask questions."
    if provider == "Google Gemini":
        res, usage = get_gemini_response(prompt, payload)
    else:
        res, usage = get_ollama_response(prompt, payload, ollama_model or default_ollama_model)
    update_token_metrics(usage)
    return res

@app.route("/")
def index():
    tok = request.args.get("token") or ""
    op = request.args.get("op") or ""
    payload = STATE["view_cache"].pop(tok, None) if tok else None
    if payload:
        STATE["summary_all"] = payload.get("summary_all") or ""
        STATE["summary_visible"] = bool(STATE["summary_all"]) and payload.get("show_summary")
        i = payload.get("article_index")
        if i is not None:
            s = payload.get("article_summary") or ""
            if s:
                STATE["summaries"][i] = s
                STATE["expanded_article"] = i
            ab = payload.get("article_bias")
            if ab:
                STATE["bias"][i] = ab
                STATE["expanded_article"] = i
            us = payload.get("article_unbiased_summary")
            if us:
                STATE["unbiased_summary"][i] = us
                STATE["expanded_article"] = i
        bs = payload.get("bias_summary")
        if bs:
            STATE["summary_bias"] = bs
        uts = payload.get("unbiased_topic_summary")
        if uts:
            STATE["unbiased_topic_summary"] = uts

    # Default fetch if empty
    if not STATE["articles"]:
        api_key = os.getenv("NEWS_API_KEY")
        if api_key:
            default_country = os.getenv("DEFAULT_COUNTRY", "us")
            # Try to get from cache first
            k = ("top", "general", default_country)
            now = time.time()
            cached = CACHE.get(k)
            if cached and now - cached[0] < CACHE_TTL_SEC:
                STATE["articles"] = cached[1]
            else:
                try:
                    arts = fetch_top_headlines("general", default_country, api_key)
                    STATE["articles"] = arts
                    if arts:
                        CACHE[k] = (now, arts)
                except Exception:
                    pass
            if STATE["articles"]:
                STATE["topic"] = "latest"
                STATE["selected_category"] = "Trending"

    return render_template("index.html", articles=STATE["articles"], topic=STATE["topic"], selected_category=STATE.get("selected_category",""), summary_all=STATE["summary_all"], summary_visible=STATE["summary_visible"], categories=CATEGORIES, bias_summary=STATE.get("summary_bias", {}), answers=STATE["answers"], summaries=STATE["summaries"], bias=STATE["bias"], unbiased=STATE["unbiased"], unbiased_summary=STATE["unbiased_summary"], unbiased_topic_summary=STATE.get("unbiased_topic_summary"), expanded_article=STATE.get("expanded_article"), error_message=request.args.get("error"), op=op, settings=SETTINGS, max_tokens=MAX_TOKENS) 

@app.route("/fetch", methods=["POST"]) 
def fetch_route():
    api_key = os.getenv("NEWS_API_KEY")
    q = request.form.get("topic", "").strip()
    if not q:
        q = "latest"
    if api_key:
        k = ("topic", q)
        now = time.time()
        cached = CACHE.get(k)
        if cached and now - cached[0] < CACHE_TTL_SEC:
            STATE["articles"] = cached[1]
        else:
            arts = fetch_news(q, api_key)
            STATE["articles"] = arts
            if arts:
                CACHE[k] = (now, arts)
        STATE["topic"] = q
        STATE["summary_all"] = ""
        STATE["summary_visible"] = False
        STATE["summary_bias"] = {}
        STATE["unbiased_topic_summary"] = ""
    return redirect(url_for("index", op="fetch"))

@app.route("/fetch_category", methods=["POST"]) 
def fetch_category_route():
    api_key = os.getenv("NEWS_API_KEY")
    cat = request.form.get("category","Trending")
    default_country = os.getenv("DEFAULT_COUNTRY", "us")
    if api_key:
        now = time.time()
        if cat == "India":
            k = ("top", "general", "in")
            cached = CACHE.get(k)
            if cached and now - cached[0] < CACHE_TTL_SEC:
                STATE["articles"] = cached[1]
            else:
                arts = fetch_top_headlines("general", "in", api_key)
                STATE["articles"] = arts
                if arts:
                    CACHE[k] = (now, arts)
        elif cat == "International":
            k = ("query", "international OR world")
            cached = CACHE.get(k)
            if cached and now - cached[0] < CACHE_TTL_SEC:
                STATE["articles"] = cached[1]
            else:
                arts = fetch_everything_query("international OR world", api_key, sort_by="publishedAt")
                STATE["articles"] = arts
                if arts:
                    CACHE[k] = (now, arts)
        elif cat == "Politics":
            k = ("query", "politics")
            cached = CACHE.get(k)
            if cached and now - cached[0] < CACHE_TTL_SEC:
                STATE["articles"] = cached[1]
            else:
                arts = fetch_everything_query("politics", api_key, sort_by="publishedAt")
                STATE["articles"] = arts
                if arts:
                    CACHE[k] = (now, arts)
        elif cat in ("Trending","General"):
            k = ("top", "general", default_country)
            cached = CACHE.get(k)
            if cached and now - cached[0] < CACHE_TTL_SEC:
                STATE["articles"] = cached[1]
            else:
                arts = fetch_top_headlines("general", default_country, api_key)
                STATE["articles"] = arts
                if arts:
                    CACHE[k] = (now, arts)
        else:
            k = ("top", cat.lower(), default_country)
            cached = CACHE.get(k)
            if cached and now - cached[0] < CACHE_TTL_SEC:
                STATE["articles"] = cached[1]
            else:
                arts = fetch_top_headlines(cat.lower(), default_country, api_key)
                STATE["articles"] = arts
                if arts:
                    CACHE[k] = (now, arts)
        STATE["topic"] = "latest"
        STATE["selected_category"] = cat
        STATE["summary_all"] = ""
        STATE["summary_visible"] = False
        STATE["summary_bias"] = {}
        STATE["unbiased_topic_summary"] = ""
    return redirect(url_for("index", op="fetch"))

@app.route("/summarize_all", methods=["POST"]) 
def summarize_all_route():
    llm_provider, ollama_model = provider_and_model()
    tpc = STATE.get("topic", "")
    s = get_cached_summary(tpc)
    if not s:
        s = summarize_all_articles(STATE["articles"], llm_provider, ollama_model, ollama_model)
        save_cached_summary(tpc, s)
    tok = f"tok_{int(time.time()*1000)}"
    STATE["view_cache"][tok] = {"summary_all": s, "show_summary": True}
    try:
        STATE["metrics"]["num_summaries_all"] += 1
    except Exception:
        pass
    save_metrics()
    try:
        add_document(s, {"type":"topic","topic":STATE.get("topic","")})
    except Exception:
        pass
    return redirect(url_for("index", token=tok))

@app.route("/summary_bias", methods=["POST"]) 
def summary_bias_route():
    llm_provider, ollama_model = provider_and_model()
    ts = STATE.get("summary_all","")
    prompt = "Analyze this topic summary for bias using simple language. State the overall bias in one short sentence, then list short bullets covering framing, selection/omission, word choice, and sensationalism. Give brief examples. Output ONLY the analysis. No conversational filler. Do not ask questions."
    if ts:
        if llm_provider == "Google Gemini":
            analysis, usage = get_gemini_response(prompt, ts)
        else:
            analysis, usage = get_ollama_response(prompt, ts, ollama_model)
        update_token_metrics(usage)
        STATE["summary_bias"]["analysis"] = analysis
        score_obj = compute_bias_score_for_text(ts, llm_provider, ollama_model, ollama_model)
        STATE["summary_bias"]["score"] = score_obj
        try:
            STATE["metrics"]["num_bias_topic"] += 1
        except Exception:
            pass
        save_metrics()
    return redirect(url_for("index"))

@app.route("/unbiased_topic_summary", methods=["POST"]) 
def unbiased_topic_summary_route():
    llm_provider, ollama_model = provider_and_model()
    ts = STATE.get("summary_all", "")
    if ts:
        prompt = "Rewrite this topic summary in neutral, unbiased language. 120-160 words, concise, factual. Output ONLY the rewritten summary. No conversational filler. Do not ask questions."
        if llm_provider == "Google Gemini":
            uts, usage = get_gemini_response(prompt, ts)
        else:
            uts, usage = get_ollama_response(prompt, ts, ollama_model)
        update_token_metrics(usage)
        STATE["unbiased_topic_summary"] = uts
        try:
            STATE["metrics"]["num_unbiased_topic"] += 1
        except Exception:
            pass
        save_metrics()
    return redirect(url_for("index"))

@app.route("/ask_global", methods=["POST"]) 
def ask_global_route():
    llm_provider, ollama_model = provider_and_model()
    q = request.form.get("question","")
    res = query(q, n_results=4)
    docs = res.get("documents", [[]])[0]
    snippets = []
    total = 0
    if STATE.get("summary_all"):
        base = STATE["summary_all"][:int(SETTINGS.get("context_limit", 1200)/2)]
        snippets.append(base)
        total += len(base)
    for d in docs:
        if not d:
            continue
        s = d.strip()
        if not s:
            continue
        if total + len(s) > SETTINGS.get("context_limit", 1200):
            s = s[: max(0, SETTINGS.get("context_limit", 1200) - total)]
        snippets.append(s)
        total += len(s)
        if total >= SETTINGS.get("context_limit", 1200):
            break
    if not snippets and STATE["articles"]:
        raw = []
        for a in STATE["articles"][:5]:
            t = a.get("title") or ""
            d = a.get("description") or ""
            c = a.get("content") or ""
            b = (t+"\n"+d+"\n"+c).strip()
            if b:
                raw.append(b)
        fb = "\n\n".join(raw)[:SETTINGS.get("context_limit", 1200)]
        if fb:
            snippets.append(fb)
    ctx = "\n\n".join(snippets)
    prompt = "Answer the question concisely using the provided context. Limit to 80-120 words. Output ONLY the answer. No conversational filler. Do not ask questions."
    payload = ctx + "\n\nQuestion: " + q
    t0 = time.time()
    if llm_provider == "Google Gemini":
        ans, usage = get_gemini_response(prompt, payload)
    else:
        ans, usage = get_ollama_response(prompt, payload, ollama_model)
    update_token_metrics(usage)
    dt_ms = (time.time() - t0) * 1000.0
    STATE["answers"]["global"] = ans
    try:
        words = len((ans or "").split())
        m = STATE["metrics"]
        m["num_global_questions"] += 1
        m["answer_latency_ms_sum"] += dt_ms
        m["answer_words_sum"] += words
        m["answer_count"] += 1
        tok_count = usage.get("total_tokens", 0) if usage else 0
        m["examples"].append({
            "type": "global",
            "question": q,
            "answer_words": words,
            "latency_ms": dt_ms,
            "tokens": tok_count
        })
        add_example({"type":"global","index":None,"question":q,"answer_words":words,"latency_ms":dt_ms, "tokens": tok_count})
    except Exception:
        pass
    save_metrics()
    return redirect(url_for("index"))

@app.route("/summarize_article/<int:i>", methods=["POST"]) 
def summarize_article_route(i):
    llm_provider, ollama_model = provider_and_model()
    art = STATE["articles"][i]
    s = get_cached_article_summary(art)
    if not s:
        s = summarize_article(art, llm_provider, ollama_model, ollama_model)
        save_cached_article_summary(art, s)
    promote_article(art)
    tok = f"tok_{int(time.time()*1000)}_0"
    STATE["view_cache"][tok] = {"article_index": 0, "article_summary": s}
    try:
        STATE["metrics"]["num_article_summaries"] += 1
    except Exception:
        pass
    save_metrics()
    try:
        add_document(s, {"type":"article","index":i,"source":art.get('source',{}).get('name')})
    except Exception:
        pass
    return redirect(url_for("index", token=tok))

@app.route("/article_bias/<int:i>", methods=["POST"]) 
def article_bias_route(i):
    llm_provider, ollama_model = provider_and_model()
    art = STATE["articles"][i]
    parts = []
    if art.get("title"):
        parts.append("Title: "+art.get("title"))
    if art.get("description"):
        parts.append("Description: "+art.get("description"))
    if art.get("content"):
        parts.append("Content: "+art.get("content"))
    payload = "\n\n".join(parts)
    prompt = "Analyze the following news article for bias. Identify any biased sentences and explain the type of bias. Output ONLY the analysis. No conversational filler. Do not ask questions."
    if payload:
        if llm_provider == "Google Gemini":
            analysis, usage = get_gemini_response(prompt, payload)
        else:
            analysis, usage = get_ollama_response(prompt, payload, ollama_model)
        update_token_metrics(usage)
        STATE["bias"][i] = {"analysis":analysis}
        score_obj = compute_bias_score_for_text(payload, llm_provider, ollama_model, ollama_model)
        STATE["bias"][i]["score"] = score_obj
        try:
            STATE["metrics"]["num_bias_article"] += 1
        except Exception:
            pass
        save_metrics()
    promote_article(art)
    return redirect(url_for("index"))

@app.route("/rewrite_article/<int:i>", methods=["POST"]) 
def rewrite_article_route(i):
    llm_provider, ollama_model = provider_and_model()
    art = STATE["articles"][i]
    parts = []
    if art.get("title"):
        parts.append("Title: "+art.get("title"))
        update_token_metrics(usage)
        STATE["unbiased"][i] = unbiased
    return redirect(url_for("index"))

@app.route("/unbiased_summary/<int:i>", methods=["POST"]) 
def unbiased_summary_route(i):
    llm_provider, ollama_model = provider_and_model()
    art = STATE["articles"][i]
    t = art.get("title") or ""
    d = art.get("description") or ""
    c = art.get("content") or ""
    raw = (t+"\n"+d+"\n"+c).strip()
    if raw:
        prompt = "Create an unbiased, neutral summary of the following article in 120-160 words. Remove biased language, framing, and sensationalism. Focus on factual content and key points. Output ONLY the summary. No conversational filler. Do not ask questions."
        if llm_provider == "Google Gemini":
            s, usage = get_gemini_response(prompt, raw)
        else:
            s, usage = get_ollama_response(prompt, raw, ollama_model)
        update_token_metrics(usage)
        STATE["unbiased_summary"][i] = s
        try:
            STATE["metrics"]["num_unbiased_article"] += 1
        except Exception:
            pass
        save_metrics()
    return redirect(url_for("index"))

@app.route("/ask_article/<int:i>", methods=["POST"]) 
def ask_article_route(i):
    llm_provider, ollama_model = provider_and_model()
    q = request.form.get("question","")
    art = STATE["articles"][i]
    t = art.get("title") or ""
    d = art.get("description") or ""
    c = art.get("content") or ""
    raw_source = STATE["unbiased"].get(i) or (t+"\n"+d+"\n"+c).strip()
    temp_id = f"temp_article_{i}_{os.getpid()}"
    chunks = []
    if raw_source:
        step = 600
        for k in range(0, len(raw_source), step):
            chunks.append(raw_source[k:k+step])
    ids = []
    if chunks:
        mds = [{"temp": temp_id, "index": i, "scope": "article"} for _ in chunks]
        ids = add_documents(chunks, mds)
    res = query(q, n_results=4, where={"temp": temp_id})
    docs = res.get("documents", [[]])[0]
    snippets = []
    total = 0
    base_summary = STATE["unbiased_summary"].get(i) or STATE["summaries"].get(i) or ""
    if base_summary:
        bs = base_summary[:int(SETTINGS.get("context_limit", 1200)/2)]
        snippets.append(bs)
        total += len(bs)
    for d0 in docs:
        if not d0:
            continue
        s0 = d0.strip()
        if not s0:
            continue
        if total + len(s0) > SETTINGS.get("context_limit", 1200):
            s0 = s0[: max(0, SETTINGS.get("context_limit", 1200) - total)]
        snippets.append(s0)
        total += len(s0)
        if total >= SETTINGS.get("context_limit", 1200):
            break
    if not snippets and raw_source:
        snippets.append(raw_source[:SETTINGS.get("context_limit", 1200)])
    ctx = "\n\n".join(snippets)
    prompt = "Answer the question concisely using the provided context. Limit to 80-120 words. Output ONLY the answer. No conversational filler. Do not ask questions."
    payload = ctx + "\n\nQuestion: " + q
    t0 = time.time()
    ans = get_gemini_response(prompt, payload) if llm_provider == "Google Gemini" else get_ollama_response(prompt, payload, ollama_model)
    dt_ms = (time.time() - t0) * 1000.0
    STATE["answers"][i] = ans
    if ids:
        try:
            delete_documents(ids)
        except Exception:
            pass
    try:
        words = len((ans or "").split())
        m = STATE["metrics"]
        m["num_article_questions"] += 1
        m["answer_latency_ms_sum"] += dt_ms
        m["answer_words_sum"] += words
        m["answer_count"] += 1
        m["examples"].append({
            "type": "article",
            "index": i,
            "question": q,
            "answer_words": words,
            "latency_ms": dt_ms
        })
        add_example({"type":"article","index":i,"question":q,"answer_words":words,"latency_ms":dt_ms})
    except Exception:
        pass
    save_metrics()
    return redirect(url_for("index"))

@app.route("/ask_global_stream")
def ask_global_stream():
    llm_provider, ollama_model = provider_and_model()
    q = request.args.get("q", "")
    res = query(q, n_results=4)
    docs = res.get("documents", [[]])[0]
    snippets = []
    total = 0
    if STATE.get("summary_all"):
        base = STATE["summary_all"][:int(SETTINGS.get("context_limit", 1200)/2)]
        snippets.append(base)
        total += len(base)
    for d in docs:
        if not d:
            continue
        s = d.strip()
        if not s:
            continue
        if total + len(s) > SETTINGS.get("context_limit", 1200):
            s = s[: max(0, SETTINGS.get("context_limit", 1200) - total)]
        snippets.append(s)
        total += len(s)
        if total >= SETTINGS.get("context_limit", 1200):
            break
    if not snippets and STATE["articles"]:
        raw = []
        for a in STATE["articles"][:5]:
            t = a.get("title") or ""
            d = a.get("description") or ""
            c = a.get("content") or ""
            b = (t+"\n"+d+"\n"+c).strip()
            if b:
                raw.append(b)
        fb = "\n\n".join(raw)[:SETTINGS.get("context_limit", 1200)]
        if fb:
            snippets.append(fb)
    ctx = "\n\n".join(snippets)
    prompt = "Answer the question concisely using the provided context. Limit to 80-120 words. Output ONLY the answer. No conversational filler. Do not ask questions."
    payload = ctx + "\n\nQuestion: " + q
    def generate():
        t0 = time.time()
        agg = []
        usage_data = {}
        def cb(u):
            usage_data.update(u)
        
        if llm_provider == "Google Gemini":
            for chunk in get_gemini_stream(prompt, payload, cb):
                agg.append(chunk)
                yield f"data: {chunk}\n\n"
        else:
            for chunk in get_ollama_stream(prompt, payload, ollama_model, cb):
                agg.append(chunk)
                yield f"data: {chunk}\n\n"
        
        update_token_metrics(usage_data)
        ans = "".join(agg)
        dt_ms = (time.time() - t0) * 1000.0
        STATE["answers"]["global"] = ans
        try:
            words = len((ans or "").split())
            m = STATE["metrics"]
            m["num_global_questions"] += 1
            m["answer_latency_ms_sum"] += dt_ms
            m["answer_words_sum"] += words
            m["answer_count"] += 1
            tok_count = usage_data.get("total_tokens", 0)
            m["examples"].append({"type":"global","question":q,"answer_words":words,"latency_ms":dt_ms, "tokens": tok_count})
            add_example({"type":"global","index":None,"question":q,"answer_words":words,"latency_ms":dt_ms, "tokens": tok_count})
        except Exception:
            pass
        save_metrics()
    return app.response_class(generate(), mimetype='text/event-stream')

@app.route("/ask_article_stream/<int:i>")
def ask_article_stream(i):
    llm_provider, ollama_model = provider_and_model()
    q = request.args.get("q", "")
    art = STATE["articles"][i]
    t = art.get("title") or ""
    d = art.get("description") or ""
    c = art.get("content") or ""
    raw_source = STATE["unbiased"].get(i) or (t+"\n"+d+"\n"+c).strip()
    temp_id = f"temp_article_{i}_{os.getpid()}"
    chunks = []
    if raw_source:
        step = 600
        for k in range(0, len(raw_source), step):
            chunks.append(raw_source[k:k+step])
    ids = []
    if chunks:
        mds = [{"temp": temp_id, "index": i, "scope": "article"} for _ in chunks]
        ids = add_documents(chunks, mds)
    res = query(q, n_results=4, where={"temp": temp_id})
    docs = res.get("documents", [[]])[0]
    snippets = []
    total = 0
    base_summary = STATE["unbiased_summary"].get(i) or STATE["summaries"].get(i) or ""
    if base_summary:
        bs = base_summary[:int(SETTINGS.get("context_limit", 1200)/2)]
        snippets.append(bs)
        total += len(bs)
    for d0 in docs:
        if not d0:
            continue
        s0 = d0.strip()
        if not s0:
            continue
        if total + len(s0) > SETTINGS.get("context_limit", 1200):
            s0 = s0[: max(0, SETTINGS.get("context_limit", 1200) - total)]
        snippets.append(s0)
        total += len(s0)
        if total >= SETTINGS.get("context_limit", 1200):
            break
    if not snippets and raw_source:
        snippets.append(raw_source[:SETTINGS.get("context_limit", 1200)])
    ctx = "\n\n".join(snippets)
    prompt = "Answer the question concisely using the provided context. Limit to 80-120 words. Output ONLY the answer. No conversational filler. Do not ask questions."
    payload = ctx + "\n\nQuestion: " + q
    def generate():
        t0 = time.time()
        agg = []
        usage_data = {}
        def cb(u):
            usage_data.update(u)

        if llm_provider == "Google Gemini":
            for chunk in get_gemini_stream(prompt, payload, cb):
                agg.append(chunk)
                yield f"data: {chunk}\n\n"
        else:
            for chunk in get_ollama_stream(prompt, payload, ollama_model, cb):
                agg.append(chunk)
                yield f"data: {chunk}\n\n"
        
        update_token_metrics(usage_data)
        ans = "".join(agg)
        dt_ms = (time.time() - t0) * 1000.0
        STATE["answers"][i] = ans
        try:
            words = len((ans or "").split())
            m = STATE["metrics"]
            m["num_article_questions"] += 1
            m["answer_latency_ms_sum"] += dt_ms
            m["answer_words_sum"] += words
            m["answer_count"] += 1
            tok_count = usage_data.get("total_tokens", 0)
            m["examples"].append({"type":"article","index":i,"question":q,"answer_words":words,"latency_ms":dt_ms, "tokens": tok_count})
            add_example({"type":"article","index":i,"question":q,"answer_words":words,"latency_ms":dt_ms, "tokens": tok_count})
        except Exception:
            pass
        save_metrics()
        if ids:
            try:
                delete_documents(ids)
            except Exception:
                pass
    return app.response_class(generate(), mimetype='text/event-stream')

JOBS = {}
JOB_ARGS = {}

def _run_job(jid, kind):
    try:
        if kind == "summarize_all":
            llm_provider, ollama_model = provider_and_model()
            res = summarize_all_articles(STATE["articles"], llm_provider, ollama_model, ollama_model)
            save_cached_summary(STATE.get("topic"), res)
            tok = f"tok_{int(time.time()*1000)}_{jid}"
            STATE["view_cache"][tok] = {"summary_all": res, "show_summary": True}
            try:
                STATE["metrics"]["num_summaries_all"] += 1
            except Exception:
                pass
            save_metrics()
            JOBS[jid] = {"status":"done", "token": tok}
        elif kind == "summary_bias":
            llm_provider, ollama_model = provider_and_model()
            ts = STATE.get("summary_all","")
            if not ts: return {}
            prompt = "Analyze this topic summary for bias using simple language. State the overall bias in one short sentence, then list short bullets covering framing, selection/omission, word choice, and sensationalism. Give brief examples. Output ONLY the analysis. No conversational filler. Do not ask questions."
            if llm_provider == "Google Gemini":
                analysis, usage = get_gemini_response(prompt, ts)
            else:
                analysis, usage = get_ollama_response(prompt, ts, ollama_model)
            update_token_metrics(usage)
            score_obj = compute_bias_score_for_text(ts, llm_provider, ollama_model, ollama_model)
            tok = f"tok_{int(time.time()*1000)}_{jid}"
            STATE["view_cache"][tok] = {"bias_summary": {"analysis": analysis, "score": score_obj}}
            try:
                STATE["metrics"]["num_bias_topic"] += 1
            except Exception:
                pass
            save_metrics()
            JOBS[jid] = {"status":"done", "token": tok}
        elif kind == "unbiased_topic_summary":
            llm_provider, ollama_model = provider_and_model()
            ts = STATE.get("summary_all","")
            if not ts: return {}
            prompt = "Rewrite this topic summary in neutral, unbiased language. 120-160 words, concise, factual. Output ONLY the rewritten summary. No conversational filler. Do not ask questions."
            if llm_provider == "Google Gemini":
                uts, usage = get_gemini_response(prompt, ts)
            else:
                uts, usage = get_ollama_response(prompt, ts, ollama_model)
            update_token_metrics(usage)
            tok = f"tok_{int(time.time()*1000)}_{jid}"
            STATE["view_cache"][tok] = {"unbiased_topic_summary": uts, "show_summary": True}
            try:
                STATE["metrics"]["num_unbiased_topic"] += 1
            except Exception:
                pass
            save_metrics()
            JOBS[jid] = {"status":"done", "token": tok}
        elif kind == "summarize_article":
            idx = JOB_ARGS.get(jid, {}).get("index")
            if isinstance(idx, int):
                llm_provider, ollama_model = provider_and_model()
                art = STATE["articles"][idx]
                s = get_cached_article_summary(art)
                if not s:
                    s = summarize_article(art, llm_provider, ollama_model, ollama_model)
                    save_cached_article_summary(art, s)
                promote_article(art)
                tok = f"tok_{int(time.time()*1000)}_{jid}"
                STATE["view_cache"][tok] = {"article_index": 0, "article_summary": s}
                try:
                    STATE["metrics"]["num_article_summaries"] += 1
                except Exception:
                    pass
                save_metrics()
                JOBS[jid] = {"status":"done", "token": tok}
            else:
                JOBS[jid] = {"status":"error"}
        elif kind == "article_bias":
            idx = JOB_ARGS.get(jid, {}).get("index")
            if isinstance(idx, int):
                llm_provider, ollama_model = provider_and_model()
                art = STATE["articles"][idx]
                parts = []
                if art.get("title"): parts.append("Title: "+art.get("title"))
                if art.get("description"): parts.append("Description: "+art.get("description"))
                if art.get("content"): parts.append("Content: "+art.get("content"))
                payload = "\n\n".join(parts)
                if not payload: return {}
                prompt = "Analyze the following news article for bias. Identify any biased sentences and explain the type of bias. Output ONLY the analysis. No conversational filler. Do not ask questions."
                if llm_provider == "Google Gemini":
                    analysis, usage = get_gemini_response(prompt, payload)
                else:
                    analysis, usage = get_ollama_response(prompt, payload, ollama_model)
                update_token_metrics(usage)
                score_obj = compute_bias_score_for_text(payload, llm_provider, ollama_model, ollama_model)
                promote_article(art)
                tok = f"tok_{int(time.time()*1000)}_{jid}"
                STATE["view_cache"][tok] = {"article_index": 0, "article_bias": {"analysis": analysis, "score": score_obj}}
                try:
                    STATE["metrics"]["num_bias_article"] += 1
                except Exception:
                    pass
                save_metrics()
                JOBS[jid] = {"status":"done", "token": tok}
            else:
                JOBS[jid] = {"status":"error"}
        elif kind == "unbiased_summary":
            idx = JOB_ARGS.get(jid, {}).get("index")
            if isinstance(idx, int):
                llm_provider, ollama_model = provider_and_model()
                art = STATE["articles"][idx]
                t = art.get("title") or ""
                d = art.get("description") or ""
                c = art.get("content") or ""
                raw = (t+"\n"+d+"\n"+c).strip()
                if not raw: return {}
                prompt = "Create an unbiased, neutral summary of the following article in 120-160 words. Remove biased language, framing, and sensationalism. Focus on factual content and key points. Output ONLY the summary. No conversational filler. Do not ask questions."
                if llm_provider == "Google Gemini":
                    s, usage = get_gemini_response(prompt, raw)
                else:
                    s, usage = get_ollama_response(prompt, raw, ollama_model)
                update_token_metrics(usage)
                tok = f"tok_{int(time.time()*1000)}_{jid}"
                STATE["view_cache"][tok] = {"article_index": idx, "article_unbiased_summary": s}
                try:
                    STATE["metrics"]["num_unbiased_article"] += 1
                except Exception:
                    pass
                save_metrics()
                JOBS[jid] = {"status":"done", "token": tok}
            else:
                JOBS[jid] = {"status":"error"}
    except Exception as e:
        print(f"Job error: {e}")
        JOBS[jid] = {"status":"error"}

@app.route("/api/models/ollama", methods=["GET"])
def list_ollama_models():
    try:
        import subprocess
        # Run 'ollama list' to get installed models
        # Output format is usually: NAME  ID  SIZE  MODIFIED
        # We'll parse the first column
        res = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        if res.returncode != 0:
            return {"models": []}
        lines = res.stdout.strip().split("\n")
        models = []
        for line in lines[1:]: # Skip header
            parts = line.split()
            if parts:
                models.append(parts[0])
        return {"models": models}
    except Exception:
        return {"models": []}

@app.route("/api/models/gemini", methods=["GET"])
def list_gemini_models():
    # Static list for now, or could fetch from library if available
    return {"models": ["gemini-1.5-flash", "gemini-pro", "gemini-1.5-pro"]}

@app.route("/api/settings", methods=["POST"])
def update_settings():
    data = request.json
    if not data:
        return {"status": "error", "message": "No data provided"}, 400
    
    p = data.get("provider")
    if p: SETTINGS["provider"] = p
    
    k = data.get("gemini_key")
    if k is not None: SETTINGS["gemini_key"] = k
    
    om = data.get("ollama_model")
    if om: SETTINGS["ollama_model"] = om
    
    gm = data.get("gemini_model")
    if gm: SETTINGS["gemini_model"] = gm

    cl = data.get("context_limit")
    if cl:
        try:
            val = int(cl)
            SETTINGS["context_limit"] = max(100, min(val, MAX_TOKENS))
        except: pass
    
    return {"status": "ok", "settings": SETTINGS}

@app.route("/summarize_all_async", methods=["POST"]) 
def summarize_all_async():
    jid = f"job_{int(time.time()*1000)}"
    JOBS[jid] = {"status":"running"}
    threading.Thread(target=_run_job, args=(jid, "summarize_all"), daemon=True).start()
    return json.dumps({"job_id": jid})

@app.route("/summary_bias_async", methods=["POST"]) 
def summary_bias_async():
    jid = f"job_{int(time.time()*1000)}"
    JOBS[jid] = {"status":"running"}
    threading.Thread(target=_run_job, args=(jid, "summary_bias"), daemon=True).start()
    return json.dumps({"job_id": jid})

@app.route("/suggest")
def suggest():
    try:
        api_key = os.getenv("NEWS_API_KEY")
        q = request.args.get("q", "").strip()
        out = []
        if api_key and q:
            try:
                res = fetch_everything_query(q, api_key, sort_by="publishedAt")
            except Exception:
                res = []
            for a in res[:5]:
                t = (a or {}).get("title") or ""
                if t:
                    out.append(t)
        return json.dumps({"suggestions": out})
    except Exception:
        return json.dumps({"suggestions": []})

@app.route("/unbiased_topic_summary_async", methods=["POST"]) 
def unbiased_topic_summary_async():
    jid = f"job_{int(time.time()*1000)}"
    JOBS[jid] = {"status":"running"}
    threading.Thread(target=_run_job, args=(jid, "unbiased_topic_summary"), daemon=True).start()
    return json.dumps({"job_id": jid})

@app.route("/summarize_article_async/<int:i>", methods=["POST"]) 
def summarize_article_async(i):
    jid = f"job_{int(time.time()*1000)}"
    JOBS[jid] = {"status":"running"}
    JOB_ARGS[jid] = {"index": i}
    threading.Thread(target=_run_job, args=(jid, "summarize_article"), daemon=True).start()
    return json.dumps({"job_id": jid})

@app.route("/article_bias_async/<int:i>", methods=["POST"]) 
def article_bias_async(i):
    jid = f"job_{int(time.time()*1000)}"
    JOBS[jid] = {"status":"running"}
    JOB_ARGS[jid] = {"index": i}
    threading.Thread(target=_run_job, args=(jid, "article_bias"), daemon=True).start()
    return json.dumps({"job_id": jid})

@app.route("/unbiased_summary_async/<int:i>", methods=["POST"]) 
def unbiased_summary_async(i):
    jid = f"job_{int(time.time()*1000)}"
    JOBS[jid] = {"status":"running"}
    JOB_ARGS[jid] = {"index": i}
    threading.Thread(target=_run_job, args=(jid, "unbiased_summary"), daemon=True).start()
    return json.dumps({"job_id": jid})

@app.route("/job/<jid>")
def job_status(jid):
    return json.dumps(JOBS.get(jid, {"status":"unknown"}))

@app.route("/metrics", methods=["GET"]) 
def metrics_page():
    load_metrics()
    m = STATE["metrics"]
    avg_latency = (m["answer_latency_ms_sum"] / m["answer_count"]) if m["answer_count"] else 0.0
    avg_words = (m["answer_words_sum"] / m["answer_count"]) if m["answer_count"] else 0.0
    ex = get_examples()
    return render_template("metrics.html", metrics=m, avg_latency=avg_latency, avg_words=avg_words, examples=ex)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000)
