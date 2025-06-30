# src/app.py
import json, os, boto3
from slack_sdk import WebClient
from slack_sdk.signature import SignatureVerifier

# ── env vars provided via serverless.yml ────────────────────────────────
SLACK_BOT_TOKEN      = os.environ["SLACK_BOT_TOKEN"]
SLACK_SIGNING_SECRET = os.environ["SLACK_SIGNING_SECRET"]
INVESTIGATOR_FN      = os.environ["INVESTIGATOR_FN"]   # logical name of the worker fn
SLACK_BOTS           = os.environ["SLACK_BOTS"] # Slackbots that we pay attention to

# ── clients ────────────────────────────────────────────────────────────
slack   = WebClient(token=SLACK_BOT_TOKEN)
verifier = SignatureVerifier(SLACK_SIGNING_SECRET)
lambda_  = boto3.client("lambda")

# ── Lambda handler ──────────────────────────────────────────────────────
def handler(event, _ctx):
    # 1) verify Slack signature
    if not verifier.is_valid_request(
        event["body"],
        {k.lower(): v for k, v in event["headers"].items()}
    ):
        return {"statusCode": 401, "body": "invalid signature"}

    payload = json.loads(event["body"])

    # 2) Slack URL-verification handshake
    if payload.get("type") == "url_verification":
        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"challenge": payload["challenge"]}),
        }

    # 3) message event
    if payload.get("type") == "event_callback":
        ev = payload["event"]

        # ignore non-bot messages
        if not ev.get("bot_id"):
            return {"statusCode": 200, "body": "skipped"}
        
        # only pay attention to important slack bots
        bot_id = ev.get("bot_id")
        bot_info_res = slack.bots_info(bot=bot_id)
        if not bot_info_res["ok"]:
            return {"statusCode": 200, "body": "failed to get bot info for bot " + bot_id}
        botname = bot_info_res["bot"]["name"]
        if botname not in SLACK_BOTS.split(","):
            return {"statusCode": 200, "body": "sender is not part of monitored slack bots"}
        
        slack.chat_postMessage(
            channel=ev["channel"],
            thread_ts=ev["ts"],
            text="Investigating… :mag:",
        )

        # fire-and-forget async invoke
        lambda_.invoke(
            FunctionName=INVESTIGATOR_FN,
            InvocationType="Event",
            Payload=json.dumps(ev),
        )

    return {"statusCode": 200, "body": "OK"}
