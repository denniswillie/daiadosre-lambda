"""Worker Lambda – Anthropic agent drives the investigation."""

import os, json, datetime, requests, statistics, math, textwrap, time
from urllib.parse import urljoin
from slack_sdk import WebClient
from collections import defaultdict
import anthropic
from supabase import create_client, Client

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
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

slack = WebClient(token=SLACK_TOKEN)

anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


# ─── Low-level helper functions ──────────────────────────────────────────────

def grafana_alert_rule(alert_title: str) -> str:
    """Return the PromQL condition string for the matching alert rule.

    Search strategy:
    Find the alert rule that exactly matches the given alert title.
    """
    hdrs = {"Authorization": f"Bearer {GRAFANA_TOKEN}"} if GRAFANA_TOKEN else {}
    rules_url = urljoin(GRAFANA_URL, "/api/v1/provisioning/alert-rules")
    res = requests.get(rules_url, headers=hdrs, timeout=10)
    res.raise_for_status()
    data = res.json()

    # API returns a list of rules directly
    rules = data if isinstance(data, list) else []
    for rule in rules:
        if rule.get("title").lower().strip() != alert_title.lower().strip():
            continue

        expression_ref_id = ""

        # Find the data entry with refId matching the condition, and find the refId of the expression.
        condition_ref = rule.get("condition")
        for data_entry in rule.get("data", []):
            if data_entry.get("refId") == condition_ref:
                expression_ref_id = data_entry.get("model", "").get("expression", "")
                break

        # Unable to find the expression refId, so skip this rule.
        if expression_ref_id == "":
            continue

        for data_entry in rule.get("data", []):
            if data_entry.get("refId") == expression_ref_id:
                expression = data_entry.get("model", {}).get("expr", "")
                if expression != "":
                    return expression
                
        raise ValueError(f"Unable to find the rule expression for the alert with the title: {alert_title}")
    
    raise ValueError("No matching alert rule found")


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
        build_meta = requests.get(f"{JENKINS_URL}/job/DaiadoPipeline/{b['number']}/api/json", auth=(JENKINS_USER, JENKINS_PAT), timeout=10).json()
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
    """
    Return a dictionary with:
        diff_path: unified diff (patch) for the commit compared to its parent.
        timestamp: Unix epoch seconds when the commit was pushed to the repo.
    """
    base_url = f"https://api.github.com/repos/{GH_REPO}/commits/{commit_sha}"

    # First, fetch commit metadata to obtain timestamp
    meta_resp = requests.get(
        base_url,
        headers={
            "Authorization": f"Bearer {GH_TOKEN}",
            "Accept": "application/vnd.github.v3+json",
        },
        timeout=10,
    )
    meta_resp.raise_for_status()
    meta_json = meta_resp.json()

    # prefer the committer date (when pushed), fall back to author date
    date_str = (
        meta_json.get("commit", {})
        .get("committer", {})
        .get("date")
        or meta_json.get("commit", {})
        .get("author", {})
        .get("date")
    )
    # Convert ISO 8601 date string to epoch seconds
    commit_ts = int(datetime.datetime.fromisoformat(date_str.replace("Z", "+00:00")).timestamp()) if date_str else None

    # Second, fetch the diff itself
    diff_resp = requests.get(
        base_url,
        headers={
            "Authorization": f"Bearer {GH_TOKEN}",
            "Accept": "application/vnd.github.v3.diff",
        },
        timeout=10,
    )
    diff_resp.raise_for_status()
    diff_text = diff_resp.text

    return {"diff_path": diff_text, "timestamp": commit_ts}


def loki_logs(service_name: str, start_ns: int, end_ns: int, limit: int = 100) -> list[tuple[int, str]]:
    """Return log lines for service_name in the window.

    The result is a chronologically sorted list of tuples:
        [(timestamp_ns, log_line), ...]
    where *timestamp_ns* is an integer nanoseconds epoch time returned by Loki.
    """
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

    entries: list[tuple[int, str]] = []
    for stream in r.json().get("data", {}).get("result", []):
        for ts_ns_str, line in stream.get("values", []):
            try:
                ts_ns = int(ts_ns_str)
            except ValueError:
                continue
            entries.append((ts_ns, line))

    # Sort by timestamp ascending and enforce limit
    entries.sort(key=lambda x: x[0])
    return entries[:limit]


def tempo_dependencies(service_name: str, start: int, end: int) -> dict:
    """Very naive Tempo search – returns a dependency graph of services in the form of {service_name: [dependent_services]}"""
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
    traces = r.json().get("traces", [])
    for trace in traces:
        r = requests.get(f"{TEMPO_URL}/api/traces/{trace.get('traceID')}", headers=hdrs, timeout=10)
        if r.status_code != 200:
            continue
        data = r.json()
        return tempo_dependency_graph(data)
    return None


def tempo_dependency_graph(data: dict) -> dict:
    # Step 1: Build a map of spanId -> (service_name, span)
    span_map = {}
    service_map = {}  # spanId -> service_name

    for batch in data["batches"]:
        service_name = None
        for attr in batch["resource"]["attributes"]:
            if attr["key"] == "service.name":
                service_name = attr["value"]["stringValue"]
                break

        for scopeSpan in batch["scopeSpans"]:
            for span in scopeSpan["spans"]:
                span_id = span["spanId"]
                span_map[span_id] = span
                service_map[span_id] = service_name

    # Step 2: Build the dependency graph
    dependency_graph = defaultdict(set)

    for span_id, span in span_map.items():
        parent_id = span.get("parentSpanId")
        if parent_id and parent_id in service_map:
            parent_service = service_map[parent_id]
            current_service = service_map[span_id]
            if parent_service != current_service:  # avoid self-links
                dependency_graph[parent_service].add(current_service)

    # Convert sets to sorted lists
    dependency_graph = {k: sorted(list(v)) for k, v in dependency_graph.items()}

    return dependency_graph


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

def github_ls(path: str = "", commit_sha: str | None = None) -> list[str]:
    """List files and directories at the given repository path.

    Parameters
    ----------
    path : str, optional
        Repository sub-path (default is root).
    commit_sha : str | None, optional
        Commit SHA or ref to list at. If omitted, lists at the HEAD of the
        default branch.

    Returns
    -------
    list[str]
        Paths (relative to repo root) that exist directly under *path* at the
        specified commit.
    """
    path = path.lstrip("/")
    url = f"https://api.github.com/repos/{GH_REPO}/contents/{path}"
    params = {"ref": commit_sha} if commit_sha else {}
    r = requests.get(
        url,
        params=params,
        headers={
            "Authorization": f"Bearer {GH_TOKEN}",
            "Accept": "application/vnd.github.v3+json",
        },
        timeout=10,
    )
    r.raise_for_status()
    data = r.json()
    if isinstance(data, list):
        return [item["path"] for item in data]
    return [data.get("path")]


def github_read_file(
    path: str,
    start_line: int = 1,
    end_line: int | None = None,
    commit_sha: str | None = None,
) -> str:
    """Read a slice of a file from GitHub at a specific commit/ref.

    Lines are 1-indexed and inclusive. If *end_line* is omitted, reads until end
    of the file. If *commit_sha* is omitted, defaults to HEAD of default branch.
    """
    if start_line < 1:
        raise ValueError("start_line must be ≥ 1")

    path = path.lstrip("/")
    url = f"https://api.github.com/repos/{GH_REPO}/contents/{path}"
    params = {"ref": commit_sha} if commit_sha else {}
    r = requests.get(
        url,
        params=params,
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
    if end_line < start_line:
        raise ValueError("Invalid line range")
    return "\n".join(lines[start_line - 1 : end_line])


def prom_service_metrics(service_name: str) -> list[str]:
    """Return a list of unique Prometheus metric names for a given job/service.

    This calls the /api/v1/series endpoint with a matcher that filters by the
    provided *service_name* (prometheus job label) and extracts distinct
    `__name__` values from the response.
    """
    hdrs = {"Authorization": f"Bearer {PROM_TOKEN}"} if PROM_TOKEN else {}
    url = urljoin(PROM_URL, "/api/v1/series")
    match_expr = f'{{__name__=~".*",job="{service_name}"}}'
    r = requests.get(
        url,
        headers=hdrs,
        params={"match[]": match_expr},
        timeout=10,
    )
    r.raise_for_status()
    js = r.json()
    if js.get("status") != "success":
        return []
    names = {item.get("__name__") for item in js.get("data", []) if item.get("__name__")}
    return sorted(names)


# ─── Tools for the anthropic agent ────────────────────────────────────────────────
tools = [
        {
            "name": "grafana_alert_rule",
            "description": "Fetch the PromQL expression for a Grafana alert rule. Returns the PromQL condition string.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "alert_title": {
                        "type": "string",
                        "description": "The title of the alert"
                    }
                },
                "required": ["alert_title"]
            }
        },
        {
            "name": "prom_query_range",
            "description": "Run a Prometheus range query. Returns list of (timestamp,value) tuples.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "expr": {
                        "type": "string",
                        "description": "The PromQL expression to run"
                    },
                    "start": {
                        "type": "number",
                        "description": "The start time in unix seconds"
                    },
                    "end": {
                        "type": "number",
                        "description": "The end time in unix seconds"
                    },
                    "step": {
                        "type": "string",
                        "description": "The step size for the query. Default value is 30s"
                    }
                },
                "required": ["expr", "start", "end"]
            }
        },
        {
            "name": "jenkins_recent_build",
            "description": "Get metadata for the most recent Jenkins build *before* incident time. Returns build details or null if none.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "incident_ts": {
                        "type": "number",
                        "description": "Incident timestamp in unix seconds used as upper bound when searching for the build"
                    }
                },
                "required": ["incident_ts"]
            }
        },
        {
            "name": "github_commit_diff",
            "description": "Return a dictionary with: diff_path: unified diff (patch) for the commit compared to its parent; timestamp: Unix epoch seconds when the commit was pushed to the repo.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "commit_sha": {
                        "type": "string",
                        "description": "The commit SHA to inspect"
                    }
                },
                "required": ["commit_sha"]
            }
        },
        {
            "name": "loki_logs",
            "description": "Fetch log lines from Loki for a service in a time window. Returns list of (timestamp_ns, log_line) tuples sorted chronologically. Returns an empty list if no logs are found.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "service_name": {"type": "string", "description": "The name of the service to fetch logs for"},
                    "start_ns": {"type": "number", "description": "Start time in nanoseconds since Unix epoch"},
                    "end_ns": {"type": "number", "description": "End time in nanoseconds since Unix epoch"},
                    "limit": {"type": "number", "description": "Maximum number of lines to return. Default value is 100"}
                },
                "required": ["service_name", "start_ns", "end_ns"]
            }
        },
        {
            "name": "tempo_dependencies",
            "description": "Search Tempo traces to list dependent services for a service in a time window. Returns a dependency graph of services in the form of {service_name: [dependent_services]}",
            "input_schema": {
                "type": "object",
                "properties": {
                    "service_name": {"type": "string", "description": "The name of the service to search traces for"},
                    "start": {"type": "number", "description": "Start time in unix seconds"},
                    "end": {"type": "number", "description": "End time in unix seconds"}
                },
                "required": ["service_name", "start", "end"]
            }
        },
        {
            "name": "correlate_series",
            "description": "Compute Pearson correlation coefficient between two equal-length time-series.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "series_a": {"type": "array", "description": "First time series as list of (timestamp,value) tuples"},
                    "series_b": {"type": "array", "description": "Second time series as list of (timestamp,value) tuples"}
                },
                "required": ["series_a", "series_b"]
            }
        },
        {
            "name": "github_ls",
            "description": "List directory contents at a given repository path and commit. If commit_sha omitted, lists at HEAD.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Repository path to list. Default value is root directory"},
                    "commit_sha": {"type": "string", "description": "Commit SHA or ref to list at. If omitted, lists at the HEAD of the default branch"}
                },
                "required": []
            }
        },
        {
            "name": "github_read_file",
            "description": "Read a slice of a file from the GitHub repository at a specific commit/ref (defaults to HEAD).",
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path of the file to read"},
                    "start_line": {"type": "number", "description": "Start line number (1-indexed, inclusive)"},
                    "end_line": {"type": "number", "description": "End line number (1-indexed, inclusive). Default value is end of file"},
                    "commit_sha": {"type": "string", "description": "Commit SHA or ref to list at. If omitted, defaults to HEAD of default branch"}
                },
                "required": ["path", "start_line"]
            }
        },
        {
            "name": "prom_service_metrics",
            "description": "Return a list of unique Prometheus metric names for a given job/service.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "service_name": {"type": "string", "description": "The name of the service to fetch metrics for"}
                },
                "required": ["service_name"]
            }
        },
]

SYSTEM_PROMPT = textwrap.dedent("""
You are an on-call Site Reliability Engineer (SRE) assistant.  Each time a production
alert fires you must lead an incident investigation and report back in clear,
actionable language.

Objectives for every investigation
1. Identify the most likely root-cause of the alert and write a concise summary
   (what happened, where, and why).
2. Build a chronological timeline that starts **when the culprit change was
   introduced** (e.g. code commit, deployment, infrastructure change) and ends
   at the exact moment the alert triggered.  Include key metric spikes,
   log/error bursts, trace anomalies, deploys, rollbacks, etc.
3. Propose one or more mitigation / remediation steps.  Be specific (e.g.
   rollback build #123, increase Redis memory "maxmemory" to 4 GiB, add retry on
   HTTP 503).

Telemetry & tooling landscape
• Prometheus – primary metrics datastore.  Use prom_query_range to inspect raw
  metric series.
• Grafana – dashboards & alert rules.  Use grafana_alert_rule to fetch the exact
  PromQL condition behind an alert.
• Loki – central log storage.  Use loki_logs to pull log lines for services and
  time ranges.
• Tempo – distributed tracing backend.  Use tempo_dependencies to understand
  call graphs and upstream/downstream service relationships.
• GitHub – source control.  Use github_ls / github_read_file to view code at any
  commit and github_commit_diff to inspect changes.
• Jenkins – deployment pipeline.  Use jenkins_recent_build to correlate build &
  deploy times with incident onset.

Operating guidelines
• Always prefer data-driven reasoning: metrics first, then logs, then traces,
  then code & deploy diffs.
• Work iteratively – fetch small windows of data rather than exhaustive dumps.
• Surface only the most relevant evidence in your report; link to raw data
  where helpful.
• If evidence is conflicting, state uncertainties explicitly and list next
  verification steps.
• When including code snippets, wrap single-line code in single backticks and multi-line code in triple backticks for proper formatting.
• Keep the tone professional and calm – you are helping engineers in the middle
  of an outage.

Output format
Reply with a single JSON object **only** – no additional commentary or markdown – in the shape:

{
  "root_cause": "<concise free-text summary of most likely cause>",
  "incident_timeline": [
    {"timestamp": "YYYY-MM-DDTHH:MM:SSZ", "description": "event description"},
    ...
  ],
  "mitigation_recommendations": [
    "actionable step one",
    "actionable step two"
  ]
}

Timestamps must be ISO-8601 in UTC.  Keep arrays reasonably short and include only relevant entries.
Do NOT output any credentials or tokens.
""")

TOOL_FN_MAP = {
    "grafana_alert_rule": grafana_alert_rule,
    "prom_query_range": prom_query_range,
    "jenkins_recent_build": jenkins_recent_build,
    "github_commit_diff": github_commit_diff,
    "loki_logs": loki_logs,
    "tempo_dependencies": tempo_dependencies,
    "correlate_series": correlate,
    "github_ls": github_ls,
    "github_read_file": github_read_file,
    "prom_service_metrics": prom_service_metrics,
}

THINKING_BLOCK_TYPE = "thinking"
TOOL_USE_BLOCK_TYPE = "tool_use"
TEXT_BLOCK_TYPE = "text"

def investigate(alert_title: str, alert_content: int, alert_ts: int) -> dict:
    '''
    Invoke the agent to investigate the incident and return the investigation result as json.
    '''
    
    user_prompt = f"""
    We have got an alert that you need to investigate:
    Alert title: {alert_title}
    Alert content: {alert_content}
    Alert timestamp: {alert_ts}
    Please investigate the incident and return the investigation result as json.
    """

    print("Investigating incident with user prompt: ", user_prompt)

    messages = [
        {"role": "user", "content": user_prompt}
    ]

    turn = 1
    while True:
        print(f"Turn {turn}:")
        turn += 1

        response = anthropic_client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=16000,
            system=SYSTEM_PROMPT,
            # Enable interleaved thinking with beta header
            extra_headers={
                "anthropic-beta": "interleaved-thinking-2025-05-14"
            },
            thinking={
                "type": "enabled",
                "budget_tokens": 10000
            },
            messages=messages,
            tools=tools,
        )

        if response.stop_reason in ["end_turn", "max_tokens", "pause_turn"]:
            print("Stop reason: ", response.stop_reason)
            print("Response content: ", response.content)
            # Verify that there's only 1 content block and that's a json object.
            if response.content[-1].type != TEXT_BLOCK_TYPE:
                raise ValueError("The agent stopped without providing a final conclusion.")
            
            # Parse the json object.
            try:
                investigation_result = json.loads(response.content[-1].text)
            except json.JSONDecodeError:
                raise ValueError("Response content is not a valid json object")
            return investigation_result
        
        if response.stop_reason not in ["tool_use"]:
            raise ValueError("Unexpected stop reason: ", response.stop_reason)

        thinking_blocks = []
        tool_use_blocks = []
        for block in response.content:
            if block.type == THINKING_BLOCK_TYPE:
                thinking_blocks.append(block)
                print(f"Thinking: {block.thinking}")
            elif block.type == TOOL_USE_BLOCK_TYPE:
                tool_use_blocks.append(block)
                print(f"Tool use: {block.name} with input {block.input}")
            elif block.type == TEXT_BLOCK_TYPE:
                print(f"Text: {block.text}")

        tool_results = []
        for tool_use_block in tool_use_blocks:
            tool_name = tool_use_block.name
            tool_input = tool_use_block.input or {}

            fn = TOOL_FN_MAP.get(tool_name)
            if fn is None:
                # For now, assume that this won't happen since there should be some kind of a check
                # on the anthropic client side.
                raise ValueError(f"Unknown tool requested by the model: {tool_name}")
            
            # Call the function with keyword expansion of inputs. Similar to the above, assume that the
            # arguments given by the model will be valid.
            print(f"Calling tool {tool_name} with input {tool_input}")
            tool_output = fn(**tool_input)
            tool_result = {
                "type": "tool_result",
                "tool_use_id": tool_use_block.id,
                "content": json.dumps(tool_output) if not isinstance(tool_output, str) else tool_output,
            }

            # Log the output for debugging
            print(f"Tool result for {tool_name}: {tool_result}")
            tool_results.append(tool_result)

        # Append new messages to the messages array.
        messages.append({
            "role": "assistant",
            "content": [block for block in response.content if block.type in [THINKING_BLOCK_TYPE, TOOL_USE_BLOCK_TYPE]]
        })
        messages.append({
            "role": "user",
            "content": tool_results
        })

# ─── Lambda handler ──────────────────────────────────────────────────────────
def handler(event, _ctx):
    event = json.loads(event) if isinstance(event, str) else event
    ch       = event["channel"]
    tts      = event["ts"]
    event = event["attachments"][0]
    alert_title    = event["title"]

    if "RESOLVED" in alert_title:
        return
    
    alert_text = event["text"]

    investigation_result = investigate(alert_title, alert_text, tts)
    print(investigation_result)

    # investigation_result = json.loads('{\n  "root_cause": "Recent code change in billing-service introduced a NullPointerException by calling .trim() on a potentially null username parameter, causing both billing-service and dependent frontend-service to return 500 errors",\n  "incident_timeline": [\n    {"timestamp": "2025-01-21T10:08:10Z", "description": "Jenkins build #49 deployed with commit 94cdbc7 containing problematic change to billing-service username handling"},\n    {"timestamp": "2025-01-21T10:08:45Z", "description": "Commit 94cdbc7 pushed - added .trim() to username parameter without null check in BillingService.java line 116"},\n    {"timestamp": "2025-01-21T10:33:10Z", "description": "First wave of 500 errors began appearing in billing-service (0.67 errors/sec) and frontend-service (0.70 errors/sec)"},\n    {"timestamp": "2025-01-21T10:34:10Z", "description": "Error rates peaked at ~0.85 errors/sec in both services, showing 99.77% correlation"},\n    {"timestamp": "2025-01-21T10:42:30Z", "description": "Error rates dropped to zero temporarily"},\n    {"timestamp": "2025-01-21T10:57:30Z", "description": "Second wave of errors began, reaching peak of 0.85 errors/sec again"},\n    {"timestamp": "2025-01-21T11:09:30Z", "description": "Errors dropped to zero again, showing cyclical pattern likely related to specific request patterns"},\n    {"timestamp": "2025-01-21T11:36:30Z", "description": "TooManyErrorResponses alert triggered as error rate reached 0.12 errors/sec (12% error rate)"}\n  ],\n  "mitigation_recommendations": [\n    "Immediately rollback Jenkins build #49 to previous stable version to restore service",\n    "Fix the null pointer exception in billing-service by adding null check: change \'String username = qp.get(\\"username\\").trim();\' to \'String username = qp.get(\\"username\\"); if (username != null) username = username.trim();\'",\n    "Add validation for required query parameters in billing-service to prevent similar issues",\n    "Implement proper error handling and input validation before calling string methods on user inputs"\n  ]\n}')

    # Go through the timeline
    for ev in investigation_result["incident_timeline"]:
        if "jenkins" in ev["description"].lower():
            ev["link"] = f"{JENKINS_URL}/job/DaiadoPipeline"
            ev["linkText"] = "JENKINS"
        elif "prometheus" in ev["description"].lower():
            ev["link"] = PROM_URL
            ev["linkText"] = "PROMETHEUS"
        elif "loki" in ev["description"].lower():
            ev["link"] = LOKI_URL
            ev["linkText"] = "LOKI"
        elif "tempo" in ev["description"].lower():
            ev["link"] = TEMPO_URL
            ev["linkText"] = "TEMPO"
    
    obj = {
        "user_id": "3df0d13f-fca4-4fd6-839a-81975488f3bb",
        "alert_title": alert_title,
        "alert_content": alert_text,
        "likely_root_cause": investigation_result["root_cause"],
        "remediation_steps": investigation_result["mitigation_recommendations"],
        "timeline": investigation_result["incident_timeline"],
    }
    print(obj)
    res = supabase.table("incidents").insert(obj).execute()
    print("result from postgres: ", res)

    # Extract the id of the inserted row. `res.data` is either a list (when multiple
    # rows are returned) or a dict (if `.single()` is chained). We handle both.
    incident_id = res.data[0].get("id")

    if incident_id is None:
        print("Warning: Unable to retrieve incident_id from Supabase response")

    slack.chat_postMessage(
        channel=ch,
        thread_ts=tts,
        text=f"Investigation has been completed. See the results at https://daiado.com/incident?id={incident_id}",
    )
