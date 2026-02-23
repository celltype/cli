"""
Notification tools: email sending via SendGrid with dry-run support.
"""

from ct.tools import registry
from ct.tools.http_client import request


@registry.register(
    name="notification.send_email",
    description="Send an email notification (dry_run=True by default logs without sending)",
    category="notification",
    parameters={
        "to": "Recipient email address",
        "subject": "Email subject line",
        "body": "Email body text",
        "from_email": "Sender email (default: from config or ct@celltype.bio)",
        "dry_run": "If True (default), only log the email without sending",
    },
    usage_guide=(
        "You need to send an email notification, typically a CRO inquiry or "
        "results summary. Always dry_run=True unless user explicitly requests sending."
    ),
)
def send_email(
    to: str,
    subject: str,
    body: str,
    from_email: str = None,
    dry_run: bool = True,
    **kwargs,
) -> dict:
    """Send an email via SendGrid, or log it in dry-run mode."""
    from datetime import datetime, timezone
    from pathlib import Path

    from ct.agent.config import Config

    config = Config.load()

    if from_email is None:
        from_email = config.get("notification.from_email", "ct@celltype.bio")

    # Ensure log directory exists
    log_dir = Path.home() / ".ct"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "sent_emails.log"

    timestamp = datetime.now(timezone.utc).isoformat()
    sent = False
    error = None

    if not dry_run:
        api_key = config.get("notification.sendgrid_api_key")
        if not api_key:
            return {
                "summary": "SendGrid API key not configured. Set it with: ct config set notification.sendgrid_api_key <key>",
                "to": to,
                "subject": subject,
                "body": body,
                "dry_run": dry_run,
                "sent": False,
                "error": "missing_api_key",
            }

        payload = {
            "personalizations": [{"to": [{"email": to}]}],
            "from": {"email": from_email},
            "subject": subject,
            "content": [{"type": "text/plain", "value": body}],
        }

        resp, req_error = request(
            "POST",
            "https://api.sendgrid.com/v3/mail/send",
            json=payload,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=30,
            retries=2,
        )
        if req_error:
            error = req_error
        else:
            sent = True

    # Log the email
    status = "DRY_RUN" if dry_run else ("SENT" if sent else f"FAILED: {error}")
    log_line = f"[{timestamp}] {status} | to={to} | subject={subject}\n"
    with open(log_file, "a") as f:
        f.write(log_line)

    if dry_run:
        summary = f"[DRY RUN] Would send email to {to}: '{subject}'"
    elif sent:
        summary = f"Email sent to {to}: '{subject}'"
    else:
        summary = f"Failed to send email to {to}: {error}"

    return {
        "summary": summary,
        "to": to,
        "subject": subject,
        "body": body,
        "from_email": from_email,
        "dry_run": dry_run,
        "sent": sent,
        "error": error,
    }
