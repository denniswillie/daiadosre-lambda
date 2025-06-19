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

llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0.2)
langchain_tools: list[Tool] = []  # Tools appended below; agent initialised afterwards

# ─── Low-level helper functions ──────────────────────────────────────────────

def grafana_alert_rule(service_name: str | None = None,
                       namespace: str | None = None,
                       alert_title: str | None = None) -> str:
    """Return the PromQL condition string for the matching alert rule.

    Search strategy:
    1. list all rules via provisioning API (requires appropriate token)
    2. try to match by exact `alert_title` (case-insensitive contains)
    3. fall back to fuzzy match on service_name / namespace in the rule title
    """
    hdrs = {"Authorization": f"Bearer {GRAFANA_TOKEN}"} if GRAFANA_TOKEN else {}
    rules_url = urljoin(GRAFANA_URL, "/api/v1/provisioning/alert-rules")
    res = requests.get(rules_url, headers=hdrs, timeout=10)
    res.raise_for_status()
    data = res.json()

    rules = data.get("data", data)  # Handle both list & dict formats
    best_match = None
    for rule in rules:
        title = (rule.get("title") or "").lower()
        if alert_title and alert_title.lower() in title:
            best_match = rule
            break
        if service_name and service_name.lower() in title:
            best_match = rule  # keep looking in case we find exact title later
    if not best_match:
        raise ValueError("No matching alert rule found")

    # Depending on Grafana version, condition may be at different path
    return (
        best_match.get("condition")
        or best_match.get("rule", {}).get("condition")
        or ""
    )


def prom_query_range(expr: str, start: int, end: int, step: str = "30s") -> list[tuple[int, float]]:
    """Run a Prometheus /query_range and return list[(ts,value)]."""
    hdrs = {"Authorization": f"Bearer {PROM_TOKEN}"} if PROM_TOKEN else {}
    url = urljoin(PROM_URL, "/api/v1/query_range")
    r = requests.get(
        url,
        headers=hdrs,
        params={"query": expr, "start": start, "end": end, "step": step},
        timeout=10,
    )
    r.raise_for_status()
    js = r.json()
    if js.get("status") != "success" or not js["data"]["result"]:
        return []
    return [(int(t), float(v)) for t, v in js["data"]["result"][0]["values"]]


def jenkins_recent_build(incident_ts: int) -> dict | None:
    """Return the most recent build *before* incident_ts for the given service.

    Assumes Jenkins job is named exactly as service_name.
    """
    api = f"{JENKINS_URL}/job/DaiadoPipeline/api/json?depth=1"
    r = requests.get(api, auth=(JENKINS_USER, JENKINS_PAT), timeout=10)
    r.raise_for_status()
    js = r.json()
    builds = js.get("builds", [])
    for b in builds:
        build_meta = requests.get(f"{b['url']}api/json", auth=(JENKINS_USER, JENKINS_PAT), timeout=10).json()
        if build_meta["timestamp"] <= incident_ts * 1000:  # Jenkins ts in ms
            sha = next(
                (
                    a["lastBuiltRevision"]["SHA1"][:7]
                    for a in build_meta.get("actions", [])
                    if "lastBuiltRevision" in a
                ),
                None,
            )
            return {
                "id": build_meta["id"],
                "number": build_meta["number"],
                "timestamp": build_meta["timestamp"],
                "status": build_meta.get("result"),
                "sha": sha,
                "url": build_meta["url"],
            }
    return None  # no suitable build


def github_commit_diff(commit_sha: str) -> dict:
    """Return diff stats against a commit's parent for a single SHA."""
    url = f"https://api.github.com/repos/{GH_REPO}/commits/{commit_sha}"
    r = requests.get(
        url,
        headers={
            "Authorization": f"Bearer {GH_TOKEN}",
            "Accept": "application/vnd.github.v3+json",
        },
        timeout=10,
    )
    r.raise_for_status()
    js = r.json()
    return {
        "commit_message": js["commit"]["message"].splitlines()[0],
        "files_changed": len(js.get("files", [])),
        "additions": sum(f["additions"] for f in js.get("files", [])),
        "deletions": sum(f["deletions"] for f in js.get("files", [])),
        "files": [f["filename"] for f in js.get("files", [])],
    }


def loki_logs(service_name: str, start_ns: int, end_ns: int, limit: int = 100) -> str:
    """Return concatenated log lines for service_name in the window."""
    hdrs = {"X-Scope-OrgID": "default"}
    query = f'{{service_name="{service_name}"}}'
    r = requests.get(
        f"{LOKI_URL}/loki/api/v1/query_range",
        params={"query": query, "start": start_ns, "end": end_ns, "limit": limit},
        headers=hdrs,
        auth=(LOKI_USER, LOKI_PASSWORD),
        timeout=10,
    )
    r.raise_for_status()
    lines: list[str] = []
    for stream in r.json().get("data", {}).get("result", []):
        lines.extend(v[1] for v in stream["values"])
    return "\n".join(lines[:limit])


def tempo_dependencies(service_name: str, start: int, end: int) -> list[str]:
    """Very naive Tempo search – returns list of called services (span.serviceName)."""
    hdrs = {"Authorization": f"Bearer {TEMPO_TOKEN}"} if TEMPO_TOKEN else {}
    # Tempo search API: /api/search?limit=... requires JSON payload.   We'll best-effort implement.
    payload = {
        "start": start * 1_000_000,  # micros
        "end": end * 1_000_000,
        "service": service_name,
        "spanLimit": 20,
    }
    r = requests.post(f"{TEMPO_URL}/api/search", json=payload, headers=hdrs, timeout=10)
    if r.status_code != 200:
        return []
    spans = r.json().get("traces", [])
    deps: set[str] = set()
    for trace in spans:
        for span in trace.get("spans", []):
            svc = span.get("process", {}).get("serviceName")
            if svc and svc != service_name:
                deps.add(svc)
    return sorted(deps)


def correlate(series_a: list[tuple[int, float]], series_b: list[tuple[int, float]]) -> float:
    """Return Pearson correlation coefficient between two aligned series."""
    if not series_a or not series_b or len(series_a) != len(series_b):
        raise ValueError("Series must be same length and non-empty")
    av, bv = [v for _, v in series_a], [v for _, v in series_b]
    ma, mb = statistics.mean(av), statistics.mean(bv)
    num = sum((ai - ma) * (bi - mb) for ai, bi in zip(av, bv))
    den = math.sqrt(sum((ai - ma) ** 2 for ai in av) * sum((bi - mb) ** 2 for bi in bv))
    return 0.0 if den == 0 else num / den


# --- GitHub repo browsing helpers ---

def github_ls(path: str = "") -> list[str]:
    """List files and directories at the given repository path (default root)."""
    # Ensure no leading slash
    path = path.lstrip("/")
    url = f"https://api.github.com/repos/{GH_REPO}/contents/{path}"
    r = requests.get(
        url,
        headers={
            "Authorization": f"Bearer {GH_TOKEN}",
            "Accept": "application/vnd.github.v3+json",
        },
        timeout=10,
    )
    r.raise_for_status()
    data = r.json()
    # If path is a file, GitHub returns a dict; for a directory, returns list
    if isinstance(data, list):
        return [item["path"] for item in data]
    return [data.get("path")]


def github_read_file(path: str, start_line: int = 1, end_line: int | None = None) -> str:
    """Read a slice of a file from GitHub.

    Lines are 1-indexed and inclusive. If `end_line` is omitted the file is read
    from `start_line` to the end.
    """
    if start_line < 1:
        raise ValueError("start_line must be ≥ 1")
    # Clean up path
    path = path.lstrip("/")
    url = f"https://api.github.com/repos/{GH_REPO}/contents/{path}"
    r = requests.get(
        url,
        headers={
            "Authorization": f"Bearer {GH_TOKEN}",
            "Accept": "application/vnd.github.v3.raw",
        },
        timeout=10,
    )
    r.raise_for_status()
    lines = r.text.splitlines()
    if end_line is None:
        end_line = len(lines)
    if end_line < start_line or end_line > len(lines):
        raise ValueError("Invalid line range")
    # Convert to 0-index slice
    return "\n".join(lines[start_line - 1 : end_line])


# ─── LangChain Tool wrappers ────────────────────────────────────────────────

langchain_tools += [
    Tool(
        name="get_alert_rule",
        func=lambda inp: grafana_alert_rule(**inp),
        description=textwrap.dedent(
            """
            Fetch the PromQL expression for a Grafana alert rule.
            Input keys: service_name, namespace, alert_title (all strings; at least one required).
            Returns the PromQL condition string.
            """
        ),
    ),
    Tool(
        name="prom_query_range",
        func=lambda inp: prom_query_range(**inp),
        description=textwrap.dedent(
            """
            Run a Prometheus range query.
            Input keys: expr (string), start (unixSeconds), end (unixSeconds), step (optional, default 30s).
            Returns list of (timestamp,value) tuples.
            """
        ),
    ),
    Tool(
        name="jenkins_recent_build",
        func=lambda inp: jenkins_recent_build(**inp),
        description=textwrap.dedent(
            """
            Get metadata for the most recent Jenkins build *before* incident time.
            Input keys: incident_ts (unixSeconds).
            Returns JSON with id, number, timestamp, status, sha, url, or null if none.
            """
        ),
    ),
    Tool(
        name="github_commit_diff",
        func=lambda inp: github_commit_diff(**inp),
        description=textwrap.dedent(
            """
            Summarise code differences for a single commit compared to its parent.
            Input keys: commit_sha (string).
            Returns counts of files/additions/deletions and list of changed files.
            """
        ),
    ),
    Tool(
        name="loki_logs",
        func=lambda inp: loki_logs(**inp),
        description=textwrap.dedent(
            """
            Fetch log lines from Loki for a service in a time window.
            Input keys: service_name (string), start_ns (unixNano), end_ns (unixNano), limit (optional int).
            Returns up to <limit> concatenated log lines.
            """
        ),
    ),
    Tool(
        name="tempo_dependencies",
        func=lambda inp: tempo_dependencies(**inp),
        description=textwrap.dedent(
            """
            Search Tempo traces for the service in a time window and list dependent services.
            Input keys: service_name (string), start (unixSeconds), end (unixSeconds).
            Returns list of service names.
            """
        ),
    ),
    Tool(
        name="correlate_series",
        func=lambda inp: correlate(**inp),
        description=textwrap.dedent(
            """
            Compute Pearson correlation coefficient between two equal-length timeseries.
            Input keys: series_a (list[(ts,val)]), series_b (list[(ts,val)]).
            Returns a float in range [-1,1].
            """
        ),
    ),
    Tool(
        name="github_ls",
        func=lambda inp: github_ls(**inp),
        description=textwrap.dedent(
            """
            List directory contents of the configured GitHub repository.
            Input keys: path (optional string, default "").
            Returns list of file/directory paths at that location.
            """
        ),
    ),
    Tool(
        name="github_read_file",
        func=lambda inp: github_read_file(**inp),
        description=textwrap.dedent(
            """
            Read a slice of a file from the GitHub repository.
            Input keys: path (string), start_line (int, 1-indexed), end_line (optional int).
            Returns the text content between those lines (inclusive).
            """
        ),
    ),
]

# Now that tools are ready, initialise the agent

agent = initialize_agent(
    tools=langchain_tools,
    llm=llm,
    agent_type="openai-functions",
    verbose=False,
)

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
