from __future__ import annotations

import logging
import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

logger = logging.getLogger(__name__)


class NotificationService:
    """Email notification service for triggered alerts.

    Reads SMTP configuration from environment variables.  When the
    required variables are not set the service silently no-ops so that
    the rest of the application is never affected.
    """

    def __init__(self) -> None:
        self.smtp_host: str = os.getenv("SMTP_HOST", "")
        self.smtp_port: int = int(os.getenv("SMTP_PORT", "587"))
        self.smtp_user: str = os.getenv("SMTP_USER", "")
        self.smtp_password: str = os.getenv("SMTP_PASSWORD", "")
        self.smtp_from: str = os.getenv("SMTP_FROM", self.smtp_user)
        self.enabled: bool = os.getenv("NOTIFICATION_EMAIL_ENABLED", "false").lower() in (
            "true",
            "1",
            "yes",
        )

    @property
    def is_configured(self) -> bool:
        """Return True when the minimum SMTP settings are present."""
        return bool(self.enabled and self.smtp_host and self.smtp_from)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def send_alert_email(self, to_email: str, alert_data: dict) -> bool:
        """Send a formatted email about a triggered alert.

        Returns True on success, False on failure (never raises).
        """
        if not self.is_configured:
            logger.debug("Email notifications not configured — skipping send.")
            return False

        if not to_email:
            return False

        subject = self._build_subject(alert_data)
        body_html = self._build_body_html(alert_data)
        body_text = self._build_body_text(alert_data)

        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = self.smtp_from
        msg["To"] = to_email
        msg.attach(MIMEText(body_text, "plain"))
        msg.attach(MIMEText(body_html, "html"))

        try:
            with smtplib.SMTP(self.smtp_host, self.smtp_port, timeout=10) as server:
                server.ehlo()
                if self.smtp_port != 25:
                    server.starttls()
                    server.ehlo()
                if self.smtp_user and self.smtp_password:
                    server.login(self.smtp_user, self.smtp_password)
                server.sendmail(self.smtp_from, [to_email], msg.as_string())
            logger.info("Alert email sent to %s for %s", to_email, alert_data.get("symbol", "?"))
            return True
        except Exception as exc:
            logger.warning("Failed to send alert email to %s: %s", to_email, exc)
            return False

    def notify_alert_triggered(self, user, alert_data: dict) -> bool:
        """Send an email notification if enabled and the user has an email.

        ``user`` can be a User model instance (with an ``email`` attribute)
        or ``None``.  Returns True when an email was successfully sent.
        """
        if not self.is_configured:
            return False

        if user is None:
            logger.debug("No user associated with alert — skipping email.")
            return False

        email = getattr(user, "email", None)
        if not email:
            logger.debug("User has no email — skipping notification.")
            return False

        return self.send_alert_email(email, alert_data)

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_subject(alert_data: dict) -> str:
        symbol = alert_data.get("symbol", "Unknown")
        alert_type = alert_data.get("alert_type")
        if alert_type:
            return f"[Stock Tracker] Smart Alert triggered for {symbol} ({alert_type})"
        condition = alert_data.get("condition", "")
        target = alert_data.get("target_price", "")
        return f"[Stock Tracker] Price Alert: {symbol} {condition} {target}"

    @staticmethod
    def _build_body_text(alert_data: dict) -> str:
        lines = ["Your stock alert has been triggered.\n"]
        for key, value in alert_data.items():
            label = key.replace("_", " ").title()
            lines.append(f"  {label}: {value}")
        lines.append("\n— Stock Tracker")
        return "\n".join(lines)

    @staticmethod
    def _build_body_html(alert_data: dict) -> str:
        rows = ""
        for key, value in alert_data.items():
            label = key.replace("_", " ").title()
            rows += f"<tr><td style='padding:4px 8px;font-weight:bold'>{label}</td><td style='padding:4px 8px'>{value}</td></tr>"
        return f"""\
<html>
<body style="font-family:Arial,sans-serif;color:#333">
<h2 style="color:#1a73e8">Stock Tracker Alert</h2>
<p>Your alert has been triggered:</p>
<table style="border-collapse:collapse;border:1px solid #ddd">{rows}</table>
<br>
<p style="color:#888;font-size:12px">You received this email because you have alerts enabled in Stock Tracker.</p>
</body>
</html>"""


notification_service = NotificationService()
