"""Push regime reports to Discord or Telegram."""
import requests

from . import config


class AlertDeliveryError(RuntimeError):
    """Delivery failed; the public error deliberately omits request details."""


def _post(channel: str, url: str, payload: dict):
    try:
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
    except requests.RequestException:
        # Transport exceptions can include bot tokens or webhook URLs.
        raise AlertDeliveryError(f"{channel} alert delivery failed.") from None


def send_discord(message: str):
    if not config.DISCORD_WEBHOOK:
        return False
    _post("Discord", config.DISCORD_WEBHOOK, {"content": message})
    return True


def send_telegram(message: str):
    if not (config.TELEGRAM_BOT_TOKEN and config.TELEGRAM_CHAT_ID):
        return False
    url = f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}/sendMessage"
    _post("Telegram", url, {"chat_id": config.TELEGRAM_CHAT_ID,
                            "text": message, "parse_mode": "Markdown"})
    return True


def notify_all(message: str):
    sent = []
    failures = 0
    for send, channel in [(send_discord, "discord"),
                          (send_telegram, "telegram")]:
        try:
            if send(message):
                sent.append(channel)
        except AlertDeliveryError:
            failures += 1
    if not sent and failures:
        raise AlertDeliveryError("All configured alert deliveries failed.")
    return sent
