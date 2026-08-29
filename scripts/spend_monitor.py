"""Spend monitor: daily email digest + threshold alerts.

Modal has no email alerting of its own. Its workspace budget is a *hard stop*
(configured in the dashboard, not the CLI) — essential as a backstop, but it
tells you nothing until it has already killed your jobs. This fills the gap in
between: a daily digest, plus an alert when month-to-date crosses a threshold.

Run it three ways:

    uv run python scripts/spend_monitor.py            # print, don't send
    uv run python scripts/spend_monitor.py --send     # print and email
    modal deploy scripts/spend_monitor.py             # daily at 16:00 UTC

Deployed, it needs two Secrets (see SETUP.md):
    auto-inference-modal-token   MODAL_TOKEN_ID, MODAL_TOKEN_SECRET
    auto-inference-resend        RESEND_API_KEY, ALERT_EMAIL_TO, ALERT_EMAIL_FROM
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import modal

# ── thresholds ───────────────────────────────────────────────────
# Deliberately low. 8xH100 is ~$31.60/hr, so a forgotten container burns
# through these fast; that is exactly what you want to hear about.
MTD_WARN_USD = 50.0
MTD_ALERT_USD = 150.0
DAILY_WARN_USD = 25.0

APP_NAME = "auto-inference-spend-monitor"
image = modal.Image.debian_slim(python_version="3.12").pip_install("modal>=1.5.5", "requests")
app = modal.App(APP_NAME, image=image)


def _f(x) -> float:
    return float(x) if isinstance(x, Decimal) else float(x or 0)


def collect() -> dict:
    """Pull month-to-date spend plus a per-object breakdown for the last day."""
    from modal import Workspace

    ws = Workspace.from_context()
    s = ws.billing.summary()

    now = datetime.now(timezone.utc)
    day_start = now - timedelta(days=1)

    by_object: list[dict] = []
    daily_total = 0.0
    try:
        for item in ws.billing.report(start=day_start, end=now):
            cost = _f(item.cost)
            daily_total += cost
            if cost > 0:
                by_object.append({
                    "object_id": item.object_id,
                    "description": item.description,
                    "environment": item.environment_name,
                    "cost_usd": round(cost, 4),
                    "by_resource": {k: round(_f(v), 4) for k, v in
                                    (item.cost_by_resource or {}).items()},
                })
    except Exception as e:
        by_object = [{"error": f"{type(e).__name__}: {e}"}]

    by_object.sort(key=lambda d: d.get("cost_usd", 0), reverse=True)

    return {
        "generated_at": now.isoformat(timespec="seconds"),
        "cycle_start": s.start.isoformat(),
        "cycle_end": s.end.isoformat(),
        "mtd_metered_usd": round(_f(s.metered_cost), 4),
        "mtd_billed_usd": round(_f(s.billed_cost), 4),
        "metered_breakdown": {k: round(_f(v), 4) for k, v in
                              (s.metered_cost_breakdown or {}).items()},
        "adjustments": {k: round(_f(v), 4) for k, v in (s.adjustments or {}).items()},
        "last_24h_usd": round(daily_total, 4),
        "last_24h_by_object": by_object[:15],
    }


def severity(d: dict) -> str:
    if d["mtd_billed_usd"] >= MTD_ALERT_USD:
        return "ALERT"
    if d["mtd_billed_usd"] >= MTD_WARN_USD or d["last_24h_usd"] >= DAILY_WARN_USD:
        return "WARN"
    return "OK"


def render(d: dict) -> tuple[str, str]:
    sev = severity(d)
    subject = (f"[{sev}] Modal spend — ${d['mtd_billed_usd']:.2f} MTD, "
               f"${d['last_24h_usd']:.2f} last 24h")

    lines = [
        f"Modal spend digest — {d['generated_at']}",
        f"Severity: {sev}",
        "",
        f"Billing cycle: {d['cycle_start'][:10]} to {d['cycle_end'][:10]}",
        f"  Month to date (billed):  ${d['mtd_billed_usd']:.2f}",
        f"  Month to date (metered): ${d['mtd_metered_usd']:.2f}",
        f"  Last 24 hours:           ${d['last_24h_usd']:.2f}",
        "",
        f"Thresholds: warn ${MTD_WARN_USD:.0f} MTD / ${DAILY_WARN_USD:.0f} daily, "
        f"alert ${MTD_ALERT_USD:.0f} MTD",
        "",
        "Metered breakdown:",
    ]
    for k, v in sorted(d["metered_breakdown"].items(), key=lambda kv: -kv[1]):
        lines.append(f"  {k:<24} ${v:.4f}")

    lines += ["", "Adjustments (credits, plan, storage):"]
    for k, v in sorted(d["adjustments"].items(), key=lambda kv: kv[1]):
        lines.append(f"  {k:<24} ${v:.4f}")

    lines += ["", "Top spenders, last 24h:"]
    if not d["last_24h_by_object"]:
        lines.append("  (nothing)")
    for o in d["last_24h_by_object"]:
        if "error" in o:
            lines.append(f"  report unavailable: {o['error']}")
            continue
        res = ", ".join(f"{k} ${v:.3f}" for k, v in
                        sorted(o["by_resource"].items(), key=lambda kv: -kv[1])[:4])
        lines.append(f"  ${o['cost_usd']:>9.4f}  {o['description'][:42]:<42} {res}")

    lines += ["", "Reference: H100 $3.95/hr (8x = $31.60/hr), H200 $4.54, B200 $6.25,",
              "           A100-80 $2.50, L40S $1.95, Volumes $0.09/GiB/month.",
              "", "Hard stop is the workspace budget in the Modal dashboard;",
              "this digest is only an early warning."]
    return subject, "\n".join(lines)


def send_email(subject: str, body: str) -> dict:
    """Send via Resend. Returns the API response, or a skip marker."""
    import requests

    key = os.environ.get("RESEND_API_KEY")
    to = os.environ.get("ALERT_EMAIL_TO")
    frm = os.environ.get("ALERT_EMAIL_FROM", "onboarding@resend.dev")
    if not key or not to:
        return {"skipped": "RESEND_API_KEY or ALERT_EMAIL_TO not set"}

    r = requests.post(
        "https://api.resend.com/emails",
        headers={"Authorization": f"Bearer {key}"},
        json={"from": frm, "to": [to], "subject": subject, "text": body},
        timeout=30,
    )
    return {"status": r.status_code, "body": r.text[:300]}


@app.function(
    schedule=modal.Cron("0 16 * * *"),   # daily 16:00 UTC
    secrets=[
        modal.Secret.from_name("auto-inference-modal-token"),
        modal.Secret.from_name("auto-inference-resend"),
    ],
    timeout=300,
)
def daily_digest() -> dict:
    d = collect()
    subject, body = render(d)
    print(subject); print(body, flush=True)
    return {"severity": severity(d), "email": send_email(subject, body), "data": d}


def main() -> int:
    ap = argparse.ArgumentParser(description="Modal spend digest")
    ap.add_argument("--send", action="store_true", help="actually send the email")
    ap.add_argument("--json", action="store_true", help="print raw JSON")
    a = ap.parse_args()

    d = collect()
    if a.json:
        print(json.dumps(d, indent=2))
        return 0

    subject, body = render(d)
    print(f"Subject: {subject}\n")
    print(body)
    if a.send:
        print("\n--- send ---")
        print(json.dumps(send_email(subject, body), indent=2))
    else:
        print("\n(dry run — pass --send to email)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
