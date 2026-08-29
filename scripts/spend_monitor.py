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
#
# These watch **metered** cost, not billed. Billed stays at $0.00 for as long
# as credits absorb usage and then jumps straight to full rate -- so alerting on
# it means silence right up to the moment money starts leaving, which is the
# opposite of early warning. Metered cost tracks real consumption from the
# first dollar.
MTD_WARN_USD = 20.0
MTD_ALERT_USD = 60.0
DAILY_WARN_USD = 15.0

# Modal Starter grants $30/month of credits. Running out is the event that
# converts usage into charges, so it gets its own alert.
MONTHLY_CREDIT_USD = 30.0
CREDIT_LOW_FRAC = 0.25          # warn with under 25% of credits left

APP_NAME = "auto-inference-spend-monitor"
image = modal.Image.debian_slim(python_version="3.12").pip_install("modal>=1.5.5", "requests")
app = modal.App(APP_NAME, image=image)


def _f(x) -> float:
    return float(x) if isinstance(x, Decimal) else float(x or 0)


# Small persistent state so each run can diff against the previous one.
_STATE = "auto-inference-spend-state"


def collect() -> dict:
    """Month-to-date spend, credit headroom, and burn since the last check.

    Deliberately does **not** rely on `billing.report()` for the alert. That
    endpoint lags (it returned zero items for a day in which ~$20 was spent)
    and is rate limited (`ResourceExhaustedError` after a couple of calls), so
    a daily threshold built on it can never fire. `billing.summary()` is
    cumulative and cheap, so the spend since the last check is just the
    difference between two readings -- which is both accurate and free.
    """
    import modal
    from modal import Workspace

    ws = Workspace.from_context()
    s = ws.billing.summary()
    now = datetime.now(timezone.utc)
    metered_now = _f(s.metered_cost)

    # Diff against the previous reading to get real burn.
    prev, burn_usd, burn_hours, burn_rate = None, None, None, None
    try:
        st = modal.Dict.from_name(_STATE, create_if_missing=True)
        prev = st.get("last")
        if prev and prev.get("cycle_start") == s.start.isoformat():
            dt_h = (now.timestamp() - float(prev["ts"])) / 3600.0
            burn_usd = round(metered_now - float(prev["metered"]), 4)
            burn_hours = round(dt_h, 2)
            burn_rate = round(burn_usd / dt_h, 3) if dt_h > 0.01 else None
        st["last"] = {"ts": now.timestamp(), "metered": metered_now,
                      "cycle_start": s.start.isoformat()}
    except Exception as e:
        prev = {"error": f"{type(e).__name__}: {e}"}

    # Per-object breakdown is best-effort only: informative when it works,
    # never load-bearing for an alert.
    by_object: list[dict] = []
    try:
        for item in ws.billing.report(start=now - timedelta(days=1), end=now):
            cost = _f(item.cost)
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
        by_object = [{"error": f"{type(e).__name__}: {e} "
                      "(rate limited or lagged; not used for alerting)"}]
    by_object.sort(key=lambda d: d.get("cost_usd", 0), reverse=True)

    credits_used = abs(_f((s.adjustments or {}).get("Credits", 0)))
    credits_left = max(0.0, MONTHLY_CREDIT_USD - credits_used)

    return {
        "generated_at": now.isoformat(timespec="seconds"),
        "cycle_start": s.start.isoformat(),
        "cycle_end": s.end.isoformat(),
        "mtd_metered_usd": round(_f(s.metered_cost), 4),
        "mtd_billed_usd": round(_f(s.billed_cost), 4),
        "credits_used_usd": round(credits_used, 4),
        "credits_left_usd": round(credits_left, 4),
        "credits_left_frac": round(credits_left / MONTHLY_CREDIT_USD, 3),
        "metered_breakdown": {k: round(_f(v), 4) for k, v in
                              (s.metered_cost_breakdown or {}).items()},
        "adjustments": {k: round(_f(v), 4) for k, v in (s.adjustments or {}).items()},
        "burn_usd": burn_usd,
        "burn_hours": burn_hours,
        "burn_rate_usd_per_hour": burn_rate,
        "first_reading": prev is None,
        "last_24h_by_object": by_object[:15],
    }


def severity(d: dict) -> str:
    """Judged on metered spend and credit headroom, not on billed cost."""
    if d["mtd_metered_usd"] >= MTD_ALERT_USD or d["credits_left_usd"] <= 0:
        return "ALERT"
    if (d["mtd_metered_usd"] >= MTD_WARN_USD
            or (d.get("burn_usd") or 0) >= DAILY_WARN_USD
            or d["credits_left_frac"] <= CREDIT_LOW_FRAC):
        return "WARN"
    return "OK"


def render(d: dict) -> tuple[str, str]:
    sev = severity(d)
    burn = (f", ${d['burn_usd']:.2f} since last check"
            if d.get("burn_usd") is not None else "")
    subject = (f"[{sev}] Modal — ${d['mtd_metered_usd']:.2f} used, "
               f"${d['credits_left_usd']:.2f} credits left{burn}")

    lines = [
        f"Modal spend digest — {d['generated_at']}",
        f"Severity: {sev}",
        "",
        f"Billing cycle: {d['cycle_start'][:10]} to {d['cycle_end'][:10]}",
        f"  Usage this month (metered):  ${d['mtd_metered_usd']:.2f}",
        f"  Credits remaining:           ${d['credits_left_usd']:.2f} "
        f"of ${MONTHLY_CREDIT_USD:.0f}  ({d['credits_left_frac']:.0%})",
        (f"  Since last check:            ${d['burn_usd']:.2f} "
         f"over {d['burn_hours']:.1f}h"
         + (f"  (${d['burn_rate_usd_per_hour']:.2f}/hr)"
            if d.get("burn_rate_usd_per_hour") else "")
         if d.get("burn_usd") is not None else
         "  Since last check:            (first reading, no baseline yet)"),
        f"  Actually charged so far:     ${d['mtd_billed_usd']:.2f}",
        "",
        f"Once credits run out, metered usage becomes real charges at full rate.",
        f"At 8xH100 ($31.60/hr) the remaining credit is "
        f"{d['credits_left_usd'] / 31.60:.1f} hours.",
        "",
        f"Thresholds: warn ${MTD_WARN_USD:.0f} metered / ${DAILY_WARN_USD:.0f} per check "
        f"/ under {CREDIT_LOW_FRAC:.0%} credits, alert ${MTD_ALERT_USD:.0f} metered",
        "",
        "Metered breakdown:",
    ]
    for k, v in sorted(d["metered_breakdown"].items(), key=lambda kv: -kv[1]):
        lines.append(f"  {k:<24} ${v:.4f}")

    lines += ["", "Adjustments (credits, plan, storage):"]
    for k, v in sorted(d["adjustments"].items(), key=lambda kv: kv[1]):
        lines.append(f"  {k:<24} ${v:.4f}")

    lines += ["", "Top spenders, last 24h (best-effort; this endpoint lags):"]
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
