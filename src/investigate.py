"""Worker Lambda – LangChain agent drives the investigation."""

import os, json, datetime, requests, statistics, math, textwrap, time
from urllib.parse import urljoin
from slack_sdk import WebClient

# ---- LangChain ---------------------------------------------------------------
from langchain.agents import initialize_agent, Tool
from langchain_openai import ChatOpenAI

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# ---- ENV ---------------------------------------------------------------------
PROM_URL     = os.getenv("PROM_URL")
LOKI_URL     = os.getenv("LOKI_URL")
GRAFANA_URL  = os.getenv("GRAFANA_URL")
JENKINS_URL  = os.getenv("JENKINS_URL")
JENKINS_USER = os.getenv("JENKINS_USER")
JENKINS_PAT  = os.getenv("JENKINS_PAT")
GH_TOKEN     = os.getenv("GITHUB_TOKEN")
GH_REPO      = os.getenv("GITHUB_REPO")
SLACK_TOKEN  = os.getenv("SLACK_BOT_TOKEN")
GRAFANA_TOKEN = os.getenv("GRAFANA_TOKEN")
TEMPO_URL = os.getenv("TEMPO_URL")
PROM_TOKEN   = os.getenv("PROM_TOKEN")
LOKI_TOKEN   = os.getenv("LOKI_TOKEN")
TEMPO_TOKEN  = os.getenv("TEMPO_TOKEN")
LOKI_USER = os.getenv("LOKI_USER")
LOKI_PASSWORD = os.getenv("LOKI_PASSWORD")

slack = WebClient(token=SLACK_TOKEN)

# ─── Low-level helpers (pure functions) ───────────────────────────────────────
def grafana_rule(alert_title: str) -> str:
    url = urljoin(GRAFANA_URL, f"/api/v1/provisioning/alert-rules")
    headers = {"Authorization": f"Bearer {GRAFANA_TOKEN}"} if GRAFANA_TOKEN else {}
    js = requests.get(url, headers=headers).json()
    
    return js["data"]["rule"]["condition"]

def prom_query(expr: str, start: int, end: int, step="30s") -> list[tuple[int,float]]:
    url = urljoin(PROM_URL, "/api/v1/query_range")
    res = requests.get(url, params={
        "query": expr, "start": start, "end": end, "step": step
    }).json()                                                    # 
    if res["status"] != "success" or not res["data"]["result"]:
        return []
    return [(int(t), float(v)) for t, v in res["data"]["result"][0]["values"]]

def loki_query(logql: str, start_ns: int, end_ns: int, lim=200) -> str:
    url = urljoin(LOKI_URL, "/loki/api/v1/query_range")
    r   = requests.get(url, params={
        "query": logql, "start": start_ns, "end": end_ns, "limit": lim
    }).json()
    lines = []
    for s in r.get("data", {}).get("result", []):
        lines += [v[1] for v in s["values"]]
    return "\n".join(lines[:20])                                 # return small excerpt

def jenkins_build(job: str) -> dict:
    api = f"{JENKINS_URL}/job/{job}/lastBuild/api/json"
    js  = requests.get(api, auth=(JENKINS_USER, JENKINS_PAT)).json()  # 
    sha = next((a["lastBuiltRevision"]["SHA1"][:7]
                for a in js.get("actions", [])
                if "lastBuiltRevision" in a), None)
    return {
        "number": js.get("number"),
        "status": js.get("result"),
        "timestamp": js.get("timestamp"),
        "sha": sha,
        "url": js.get("url"),
    }

def github_diff(base_sha: str, head_sha: str) -> dict:
    url = f"https://api.github.com/repos/{GH_REPO}/compare/{base_sha}...{head_sha}"
    js  = requests.get(url, headers={
        "Authorization": f"Bearer {GH_TOKEN}",
        "Accept": "application/vnd.github.v3+json",
    }).json()                                                    # 
    return {
        "files_changed": len(js.get("files", [])),
        "additions":     sum(f["additions"]  for f in js.get("files", [])),
        "deletions":     sum(f["deletions"]  for f in js.get("files", [])),
        "titles":        [c["commit"]["message"].splitlines()[0]
                          for c in js.get("commits", [])][:5],
    }

# simple numeric correlation helper (tool not exposed to LLM)
def _pearson(a: list[tuple[int,float]], b: list[tuple[int,float]]) -> float:
    if not a or not b or len(a) != len(b): return 0
    av, bv = [v for _, v in a], [v for _, v in b]
    ma, mb = statistics.mean(av), statistics.mean(bv)
    num = sum((ai-ma)*(bi-mb) for ai, bi in zip(av, bv))
    den = math.sqrt(sum((ai-ma)**2 for ai in av) * sum((bi-mb)**2 for bi in bv))
    return 0 if den == 0 else num / den

# ── helper: extract Grafana metadata from the Slack alert text ──────────
def _parse_alert_labels(txt: str) -> dict:
    """
    Very simple parser that looks for lines like
      - __alert_rule_uid__ = eeotmqg3pj75sa
      - service = billing-service
    Adapt to match your exact message format.
    """
    res = {"rule_uid": None, "labels": {}}
    for line in txt.splitlines():
        if "alert_rule_uid" in line or line.strip().startswith("uid ="):
            res["rule_uid"] = line.split("=", 1)[1].strip()
        if line.lstrip().startswith("-") and "=" in line:
            k, v = [p.strip() for p in line.lstrip("- ").split("=", 1)]
            res["labels"][k] = v
    return res

# ─── LangChain Tools ──────────────────────────────────────────────────────────
'''

List of tools I need to build:
- 
'''
tools = [
    Tool(
        name="prom_query",
        func=lambda inp: prom_query(**inp),
        description=textwrap.dedent("""\
            Run a Prometheus range query.
            Input keys: expr (string PromQL), start (unixSeconds), end (unixSeconds), step (optional).
            Returns list of (ts,value). Keep queries narrow—15-30 min window—and short expressions.
        """)
    ),
    Tool(
        name="loki_query",
        func=lambda inp: loki_query(**inp),
        description=textwrap.dedent("""\
            Fetch log lines from Loki.
            Input keys: logql (string), start_ns (unixNano), end_ns (unixNano), lim (optional).
            Returns plain-text snippet (≤20 lines). Prefer filters that match errors/warnings only.
        """)
    ),
    Tool(
        name="jenkins_build",
        func=lambda job: jenkins_build(job),
        description="Get last Jenkins build metadata for the given job name."
    ),
    Tool(
        name="github_diff",
        func=lambda inp: github_diff(**inp),
        description="Summarise diff between two commit SHAs in the configured GitHub repo."
    )
]

# llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0.2)
# agent = initialize_agent(
#     tools=tools,
#     llm=llm,
#     agent_type="openai-functions",  # forces JSON tool calls
#     verbose=False,
# )

# ─── Lambda handler ──────────────────────────────────────────────────────────
def handler(event, _ctx):
    payload = event if isinstance(event, dict) else json.loads(event)
    ch       = payload["channel"]
    tts      = payload["ts"]

    # Test connectivity and permissions to Grafana, Loki, Prometheus, Tempo, Jenkins, GitHub.
    errors = []
    print("\n=== Starting connectivity tests ===")

    # Test GitHub - check if can read file in the repository, e.g., billing-service/src/main/java/BillingService.java
    print("\n-> Testing GitHub API...")
    try:
        github_headers = {
            "Authorization": f"Bearer {GH_TOKEN}",
            # We want the raw file contents, not the base-64 JSON wrapper
            "Accept": "application/vnd.github.v3.raw",
        }
        r = requests.get(
            f"https://api.github.com/repos/{GH_REPO}/contents/billing-service/src/main/java/BillingService.java",
            headers=github_headers,
            timeout=5,
        )
        if r.status_code != 200:
            errors.append(f"GitHub: HTTP {r.status_code}")
            print(f"✗ GitHub error: HTTP {r.status_code}")
        else:
            print(f"✓ GitHub: Successfully fetched file ({len(r.text)} bytes)")
    except Exception as e:
        errors.append(f"GitHub: {e}")
        print(f"✗ GitHub error: {e}")

    # Test Prometheus - check if can get timeseries data given expression, e.g., rate(billing_requests_total{job="billing-service"}[5m])
    print("\n-> Testing Prometheus API...")
    try:
        expr = 'rate(billing_requests_total{job="billing-service"}[5m])'
        now = int(time.time())
        start = now - 300  # last 5 minutes
        prom_headers = {"Authorization": f"Bearer {PROM_TOKEN}"} if PROM_TOKEN else {}
        print(f"  Query: {expr}")
        r = requests.get(
            f"{PROM_URL}/api/v1/query_range",
            params={"query": expr, "start": start, "end": now, "step": "30s"},
            headers=prom_headers,
            timeout=5,
        )
        js = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
        if r.status_code != 200 or js.get("status") != "success":
            errors.append(f"Prometheus: HTTP {r.status_code}")
            print(f"✗ Prometheus error: HTTP {r.status_code}")
        else:
            results = js.get("data", {}).get("result", [])
            print(f"✓ Prometheus: Got {len(results)} series")
    except Exception as e:
        errors.append(f"Prometheus: {e}")
        print(f"✗ Prometheus error: {e}")

    # Test Loki - check if can get logs, e.g., get logs from namespace "default" in the last 30 minutes
    print("\n-> Testing Loki API...")
    try:
        now_ns   = int(time.time() * 1_000_000_000)
        start_ns = now_ns - 30 * 60 * 1_000_000_000  # last 30 min
        loki_headers = {"X-Scope-OrgID": "default"}
        query = '{namespace="default"} |= "error"'
        print(f"  Query: {query}")
        r = requests.get(
            f"{LOKI_URL}/loki/api/v1/query_range",
            params={
                "query": query,
                "start": start_ns,
                "end": now_ns,
                "limit": 100
            },
            headers=loki_headers,
            auth=(LOKI_USER, LOKI_PASSWORD),
            timeout=5,
        )
        if r.status_code != 200:
            errors.append(f"Loki: HTTP {r.status_code}")
            print(f"✗ Loki error: HTTP {r.status_code}")
        else:
            js = r.json()
            streams = js.get("data", {}).get("result", [])
            print(f"✓ Loki: Found {len(streams)} streams with logs")
    except Exception as e:
        errors.append(f"Loki: {e}")
        print(f"✗ Loki error: {e}")

    # Test Tempo - simple readiness or auth check
    print("\n-> Testing Tempo API...")
    try:
        tempo_headers = {"Authorization": f"Bearer {TEMPO_TOKEN}"} if TEMPO_TOKEN else {}
        r = requests.get(f"{TEMPO_URL}/ready", headers=tempo_headers, timeout=5)
        if r.status_code != 200:
            errors.append(f"Tempo: HTTP {r.status_code}")
            print(f"✗ Tempo error: HTTP {r.status_code}")
        else:
            print("✓ Tempo: Ready check passed")
    except Exception as e:
        errors.append(f"Tempo: {e}")
        print(f"✗ Tempo error: {e}")

    # Test Jenkins - check if can get the most recent build information from DaiadoPipeline pipeline.
    print("\n-> Testing Jenkins API...")
    try:
        r = requests.get(
            f"{JENKINS_URL}/job/DaiadoPipeline/lastBuild/api/json",
            auth=(JENKINS_USER, JENKINS_PAT),
            timeout=5,
        )
        if r.status_code != 200:
            errors.append(f"Jenkins: HTTP {r.status_code}")
            print(f"✗ Jenkins error: HTTP {r.status_code}")
        else:
            js = r.json()
            print(f"✓ Jenkins: Found build #{js.get('number')} ({js.get('result')})")
    except Exception as e:
        errors.append(f"Jenkins: {e}")
        print(f"✗ Jenkins error: {e}")

    # Test Grafana - get the alert rule given its title (service + namespace), ensuring token is used
    print("\n-> Testing Grafana API...")
    try:
        grafana_headers = {}
        if GRAFANA_TOKEN:
            grafana_headers = {"Authorization": f"Bearer {GRAFANA_TOKEN}"}
            print(f"  Using auth token: {GRAFANA_TOKEN[:5]}...")
        print(f"  Headers: {grafana_headers}")
        # Retrieve all alert rules then just grab the first one as a sanity check
        r = requests.get(f"{GRAFANA_URL}/api/v1/provisioning/alert-rules", headers=grafana_headers, timeout=5)
        print(f"  Response status: {r.status_code}")
        print(f"  Response headers: {dict(r.headers)}")
        if r.status_code != 200:
            errors.append(f"Grafana: HTTP {r.status_code}")
            print(f"✗ Grafana error: HTTP {r.status_code}")
            print(f"  Response body: {r.text[:500]}")  # Print first 500 chars of error response
        else:
            js = r.json()
            print(f"  Response JSON: {js}")  # Debug the actual response structure
            # Handle both possible response formats
            rules = []
            if isinstance(js, dict):
                rules = js.get("data", [])
            elif isinstance(js, list):
                rules = js
            print(f"✓ Grafana: Found {len(rules)} alert rules")
    except Exception as e:
        errors.append(f"Grafana: {e}")
        print(f"✗ Grafana error: {e}")

    print("\n=== Connectivity test summary ===")
    if errors:
        print("❌ Some tests failed:")
        for error in errors:
            print(f"  • {error}")
        slack.chat_postMessage(
            channel=ch,
            thread_ts=tts,
            text=":x: Connectivity check failed:\n" + "\n".join(errors),
        )
        return {"statusCode": 500, "body": "Connectivity check failed: " + "; ".join(errors)}
    else:
        print("✅ All connectivity tests passed!")

    slack.chat_postMessage(
        channel=ch,
        thread_ts=tts,
        text="hello world!",
    )
