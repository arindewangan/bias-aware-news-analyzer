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
        "examples": []
    }
}

def _db_conn():
    return sqlite3.connect(DB_PATH)

def init_db():
    conn = _db_conn()
    cur = conn.cursor()
    cur.execute("CREATE TABLE IF NOT EXISTS metrics_summary (id INTEGER PRIMARY KEY, data TEXT)")
    cur.execute("CREATE TABLE IF NOT EXISTS metrics_examples (id INTEGER PRIMARY KEY AUTOINCREMENT, type TEXT, idx INTEGER, question TEXT, answer_words INTEGER, latency_ms REAL)")
    cur.execute("CREATE TABLE IF NOT EXISTS topic_summaries (topic TEXT PRIMARY KEY, summary TEXT, created_at INTEGER)")
    cur.execute("CREATE TABLE IF NOT EXISTS article_summaries (article_id TEXT PRIMARY KEY, summary TEXT, created_at INTEGER)")
    conn.commit()
    conn.close()

def load_metrics():
    try:
        init_db()
        conn = _db_conn()
        cur = conn.cursor()
        cur.execute("SELECT data FROM metrics_summary WHERE id=1")
        row = cur.fetchone()
        if row and row[0]:
            data = json.loads(row[0])
            STATE["metrics"] = data
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
        cur.execute("INSERT INTO metrics_examples (type, idx, question, answer_words, latency_ms) VALUES (?,?,?,?,?)", (ex.get("type"), ex.get("index"), ex.get("question"), ex.get("answer_words"), ex.get("latency_ms")))
        conn.commit()
        conn.close()
    except Exception:
        pass

def get_examples():
    try:
        conn = _db_conn()
        cur = conn.cursor()
        cur.execute("SELECT type, idx, question, answer_words, latency_ms FROM metrics_examples ORDER BY id DESC LIMIT 100")
        rows = cur.fetchall()
        conn.close()
        out = []
        for t,i,q,w,l in rows:
            out.append({"type": t, "index": i, "question": q, "answer_words": w, "latency_ms": l})
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

def provider_and_model():
    env_provider = os.getenv("LLM_PROVIDER", "gemini").strip().lower()
    llm_provider = "Ollama" if env_provider in ("ollama", "local") else "Google Gemini"
    default_ollama_model = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
    return llm_provider, default_ollama_model

def compute_bias_score_for_text(text, provider, ollama_model, default_ollama_model):
    payload = (text or "").strip()
    prompt = "Assess overall bias in this text. Respond ONLY with a JSON object using keys 'score' (integer 0-100) and 'rationale' (string). No extra text or markdown."
    if provider == "Google Gemini":
        raw = get_gemini_response(prompt, payload)
    else:
        raw = get_ollama_response(prompt, payload, ollama_model or default_ollama_model)
    try:
        import json as _json
        data = _json.loads(raw)
        if isinstance(data, dict) and "score" in data and "rationale" in data:
            try:
                sc = int(float(data.get("score")))
                sc = max(0, min(100, sc))
            except Exception:
                sc = None
            return {"score": sc, "rationale": data.get("rationale")}
    except Exception:
        pass
    try:
        import re as _re
        m = _re.search(r"score\s*[:=]\s*(\d{1,3})", raw, flags=_re.IGNORECASE)
        if not m:
            m = _re.search(r"(\d{1,3})\s*%", raw)
        sc = None
        if m:
            sc = int(m.group(1))
            sc = max(0, min(100, sc))
        return {"score": sc, "rationale": raw}
    except Exception:
        return {"score": None, "rationale": raw}

def summarize_article(article, provider, ollama_model, default_ollama_model):
    parts = []
    t = article.get("title")
    d = article.get("description")
    c = article.get("content")
    if t:
        parts.append("Title: " + t)
    if d:
        parts.append("Description: " + d)
    if c:
        parts.append("Content: " + c)
    payload = "\n\n".join(parts).strip()
    if not payload:
        return ""
    prompt = "Summarize the main points, perspective, and context in this article in 120-160 words."
    if provider == "Google Gemini":
        return get_gemini_response(prompt, payload)
    return get_ollama_response(prompt, payload, ollama_model or default_ollama_model)

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
    prompt = "Summarize the overall topic and key points across these articles in neutral tone. Provide a concise overview and exactly 5 bullet key takeaways. Do not ask questions, do not request more input, and do not add suggestions."
    if provider == "Google Gemini":
        return get_gemini_response(prompt, payload)
    return get_ollama_response(prompt, payload, ollama_model or default_ollama_model)

@app.route("/", methods=["GET"]) 
def index():
    api_key = os.getenv("NEWS_API_KEY")
    default_country = os.getenv("DEFAULT_COUNTRY", "us")
    categories = ["Trending","General","Business","Entertainment","Health","Science","Sports","Technology","Politics","International","India"]
    if not STATE["articles"] and api_key:
        ensure_collection()
        k = ("top", "general", default_country)
        now = time.time()
        cached = CACHE.get(k)
        if cached and now - cached[0] < CACHE_TTL_SEC:
            STATE["articles"] = cached[1]
        else:
            arts = fetch_top_headlines("general", default_country, api_key)
            STATE["articles"] = arts
            CACHE[k] = (now, arts)
        STATE["topic"] = "latest"
    STATE["summary_all"] = ""
    STATE["summary_visible"] = False
    STATE["answers"] = {}
    STATE["summaries"] = {}
    STATE["bias"] = {}
    STATE["unbiased"] = {}
    STATE["unbiased_summary"] = {}
    STATE["expanded_article"] = None
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
    return render_template("index.html", articles=STATE["articles"], topic=STATE["topic"], selected_category=STATE.get("selected_category",""), summary_all=STATE["summary_all"], summary_visible=STATE["summary_visible"], categories=categories, bias_summary=STATE.get("summary_bias", {}), answers=STATE["answers"], summaries=STATE["summaries"], bias=STATE["bias"], unbiased=STATE["unbiased"], unbiased_summary=STATE["unbiased_summary"], unbiased_topic_summary=STATE.get("unbiased_topic_summary"), expanded_article=STATE.get("expanded_article"), error_message=request.args.get("error"), op=op) 

@app.route("/fetch", methods=["POST"]) 
def fetch_route():
    api_key = os.getenv("NEWS_API_KEY")
    q = request.form.get("topic","artificial intelligence")
    if api_key:
        k = ("topic", q)
        now = time.time()
        cached = CACHE.get(k)
        if cached and now - cached[0] < CACHE_TTL_SEC:
            STATE["articles"] = cached[1]
        else:
            arts = fetch_news(q, api_key)
            STATE["articles"] = arts
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
                CACHE[k] = (now, arts)
        elif cat == "International":
            k = ("query", "international OR world")
            cached = CACHE.get(k)
            if cached and now - cached[0] < CACHE_TTL_SEC:
                STATE["articles"] = cached[1]
            else:
                arts = fetch_everything_query("international OR world", api_key, sort_by="publishedAt")
                STATE["articles"] = arts
                CACHE[k] = (now, arts)
        elif cat == "Politics":
            k = ("query", "politics")
            cached = CACHE.get(k)
            if cached and now - cached[0] < CACHE_TTL_SEC:
                STATE["articles"] = cached[1]
            else:
                arts = fetch_everything_query("politics", api_key, sort_by="publishedAt")
                STATE["articles"] = arts
                CACHE[k] = (now, arts)
        elif cat in ("Trending","General"):
            k = ("top", "general", default_country)
            cached = CACHE.get(k)
            if cached and now - cached[0] < CACHE_TTL_SEC:
                STATE["articles"] = cached[1]
            else:
                arts = fetch_top_headlines("general", default_country, api_key)
                STATE["articles"] = arts
                CACHE[k] = (now, arts)
        else:
            k = ("top", cat.lower(), default_country)
            cached = CACHE.get(k)
            if cached and now - cached[0] < CACHE_TTL_SEC:
                STATE["articles"] = cached[1]
            else:
                arts = fetch_top_headlines(cat.lower(), default_country, api_key)
                STATE["articles"] = arts
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
    prompt = "Analyze this topic summary for bias using simple language. State the overall bias in one short sentence, then list short bullets covering framing, selection/omission, word choice, and sensationalism. Give brief examples. Do not ask questions."
    if ts:
        analysis = get_gemini_response(prompt, ts) if llm_provider == "Google Gemini" else get_ollama_response(prompt, ts, ollama_model)
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
        prompt = "Rewrite this topic summary in neutral, unbiased language. 120-160 words, concise, factual, no suggestions or questions."
        uts = get_gemini_response(prompt, ts) if llm_provider == "Google Gemini" else get_ollama_response(prompt, ts, ollama_model)
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
        base = STATE["summary_all"][:600]
        snippets.append(base)
        total += len(base)
    for d in docs:
        if not d:
            continue
        s = d.strip()
        if not s:
            continue
        if total + len(s) > 1200:
            s = s[: max(0, 1200 - total)]
        snippets.append(s)
        total += len(s)
        if total >= 1200:
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
        fb = "\n\n".join(raw)[:1200]
        if fb:
            snippets.append(fb)
    ctx = "\n\n".join(snippets)
    prompt = "Answer the question concisely using the provided context. Limit to 80-120 words."
    payload = ctx + "\n\nQuestion: " + q
    t0 = time.time()
    ans = get_gemini_response(prompt, payload) if llm_provider == "Google Gemini" else get_ollama_response(prompt, payload, ollama_model)
    dt_ms = (time.time() - t0) * 1000.0
    STATE["answers"]["global"] = ans
    try:
        words = len((ans or "").split())
        m = STATE["metrics"]
        m["num_global_questions"] += 1
        m["answer_latency_ms_sum"] += dt_ms
        m["answer_words_sum"] += words
        m["answer_count"] += 1
        m["examples"].append({
            "type": "global",
            "question": q,
            "answer_words": words,
            "latency_ms": dt_ms
        })
        add_example({"type":"global","index":None,"question":q,"answer_words":words,"latency_ms":dt_ms})
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
    tok = f"tok_{int(time.time()*1000)}_{i}"
    STATE["view_cache"][tok] = {"article_index": i, "article_summary": s}
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
    prompt = "Analyze the following news article for bias. Identify any biased sentences and explain the type of bias."
    if payload:
        analysis = get_gemini_response(prompt, payload) if llm_provider == "Google Gemini" else get_ollama_response(prompt, payload, ollama_model)
        STATE["bias"][i] = {"analysis":analysis}
        score_obj = compute_bias_score_for_text(payload, llm_provider, ollama_model, ollama_model)
        STATE["bias"][i]["score"] = score_obj
        try:
            STATE["metrics"]["num_bias_article"] += 1
        except Exception:
            pass
        save_metrics()
    STATE["expanded_article"] = i
    return redirect(url_for("index"))

@app.route("/rewrite_article/<int:i>", methods=["POST"]) 
def rewrite_article_route(i):
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
    prompt = "Rewrite the following news article in a neutral and objective tone."
    if payload:
        unbiased = get_gemini_response(prompt, payload) if llm_provider == "Google Gemini" else get_ollama_response(prompt, payload, ollama_model)
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
        prompt = "Create an unbiased, neutral summary of the following article in 120-160 words. Remove biased language, framing, and sensationalism. Focus on factual content and key points."
        s = get_gemini_response(prompt, raw) if llm_provider == "Google Gemini" else get_ollama_response(prompt, raw, ollama_model)
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
        bs = base_summary[:600]
        snippets.append(bs)
        total += len(bs)
    for d0 in docs:
        if not d0:
            continue
        s0 = d0.strip()
        if not s0:
            continue
        if total + len(s0) > 1200:
            s0 = s0[: max(0, 1200 - total)]
        snippets.append(s0)
        total += len(s0)
        if total >= 1200:
            break
    if not snippets and raw_source:
        snippets.append(raw_source[:1200])
    ctx = "\n\n".join(snippets)
    prompt = "Answer the question concisely using the provided context. Limit to 80-120 words."
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
        base = STATE["summary_all"][:600]
        snippets.append(base)
        total += len(base)
    for d in docs:
        if not d:
            continue
        s = d.strip()
        if not s:
            continue
        if total + len(s) > 1200:
            s = s[: max(0, 1200 - total)]
        snippets.append(s)
        total += len(s)
        if total >= 1200:
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
        fb = "\n\n".join(raw)[:1200]
        if fb:
            snippets.append(fb)
    ctx = "\n\n".join(snippets)
    prompt = "Answer the question concisely using the provided context. Limit to 80-120 words."
    payload = ctx + "\n\nQuestion: " + q
    def generate():
        t0 = time.time()
        agg = []
        if llm_provider == "Google Gemini":
            for chunk in get_gemini_stream(prompt, payload):
                agg.append(chunk)
                yield f"data: {chunk}\n\n"
        else:
            for chunk in get_ollama_stream(prompt, payload, ollama_model):
                agg.append(chunk)
                yield f"data: {chunk}\n\n"
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
            m["examples"].append({"type":"global","question":q,"answer_words":words,"latency_ms":dt_ms})
            add_example({"type":"global","index":None,"question":q,"answer_words":words,"latency_ms":dt_ms})
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
        bs = base_summary[:600]
        snippets.append(bs)
        total += len(bs)
    for d0 in docs:
        if not d0:
            continue
        s0 = d0.strip()
        if not s0:
            continue
        if total + len(s0) > 1200:
            s0 = s0[: max(0, 1200 - total)]
        snippets.append(s0)
        total += len(s0)
        if total >= 1200:
            break
    if not snippets and raw_source:
        snippets.append(raw_source[:1200])
    ctx = "\n\n".join(snippets)
    prompt = "Answer the question concisely using the provided context. Limit to 80-120 words."
    payload = ctx + "\n\nQuestion: " + q
    def generate():
        t0 = time.time()
        agg = []
        if llm_provider == "Google Gemini":
            for chunk in get_gemini_stream(prompt, payload):
                agg.append(chunk)
                yield f"data: {chunk}\n\n"
        else:
            for chunk in get_ollama_stream(prompt, payload, ollama_model):
                agg.append(chunk)
                yield f"data: {chunk}\n\n"
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
            m["examples"].append({"type":"article","index":i,"question":q,"answer_words":words,"latency_ms":dt_ms})
            add_example({"type":"article","index":i,"question":q,"answer_words":words,"latency_ms":dt_ms})
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
            tpc = STATE.get("topic", "")
            s = get_cached_summary(tpc)
            if not s:
                s = summarize_all_articles(STATE["articles"], llm_provider, ollama_model, ollama_model)
                save_cached_summary(tpc, s)
            tok = f"tok_{int(time.time()*1000)}_{jid}"
            STATE["view_cache"][tok] = {"summary_all": s, "show_summary": True}
            try:
                STATE["metrics"]["num_summaries_all"] += 1
            except Exception:
                pass
            save_metrics()
            JOBS[jid] = {"status":"done", "token": tok}
        elif kind == "summary_bias":
            llm_provider, ollama_model = provider_and_model()
            ts = STATE.get("summary_all", "")
            if ts:
                prompt = "Analyze this topic summary for bias using simple language. State the overall bias in one short sentence, then list short bullets covering framing, selection/omission, word choice, and sensationalism. Give brief examples. Do not ask questions."
                analysis = get_gemini_response(prompt, ts) if llm_provider == "Google Gemini" else get_ollama_response(prompt, ts, ollama_model)
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
            ts = STATE.get("summary_all", "")
            tok = None
            if ts:
                prompt = "Rewrite this topic summary in neutral, unbiased language. 120-160 words, concise, factual, no suggestions or questions."
                uts = get_gemini_response(prompt, ts) if llm_provider == "Google Gemini" else get_ollama_response(prompt, ts, ollama_model)
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
                tok = f"tok_{int(time.time()*1000)}_{jid}"
                STATE["view_cache"][tok] = {"article_index": idx, "article_summary": s}
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
                if art.get("title"):
                    parts.append("Title: "+art.get("title"))
                if art.get("description"):
                    parts.append("Description: "+art.get("description"))
                if art.get("content"):
                    parts.append("Content: "+art.get("content"))
                payload = "\n\n".join(parts)
                tok = None
                if payload:
                    prompt = "Analyze the following news article for bias. Identify any biased sentences and explain the type of bias."
                    analysis = get_gemini_response(prompt, payload) if llm_provider == "Google Gemini" else get_ollama_response(prompt, payload, ollama_model)
                    score_obj = compute_bias_score_for_text(payload, llm_provider, ollama_model, ollama_model)
                    tok = f"tok_{int(time.time()*1000)}_{jid}"
                    STATE["view_cache"][tok] = {"article_index": idx, "article_bias": {"analysis": analysis, "score": score_obj}}
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
                tok = None
                if raw:
                    prompt = "Create an unbiased, neutral summary of the following article in 120-160 words. Remove biased language, framing, and sensationalism. Focus on factual content and key points."
                    s = get_gemini_response(prompt, raw) if llm_provider == "Google Gemini" else get_ollama_response(prompt, raw, ollama_model)
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
    except Exception:
        JOBS[jid] = {"status":"error"}

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