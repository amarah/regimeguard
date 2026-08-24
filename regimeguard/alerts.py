"""Push regime reports to Discord or Telegram."""
import requests

from . import config


def send_discord(message: str):
    if not config.DISCORD_WEBHOOK:
        return False
    requests.post(config.DISCORD_WEBHOOK, json={"content": message}, timeout=10)
    return True


def send_telegram(message: str):
    if not (config.TELEGRAM_BOT_TOKEN and config.TELEGRAM_CHAT_ID):
        return False
    url = f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}/sendMessage"
    requests.post(url, json={"chat_id": config.TELEGRAM_CHAT_ID,
                             "text": message, "parse_mode": "Markdown"}, timeout=10)
    return True


def notify_all(message: str):
    sent = [ch for fn, ch in [(send_discord, "discord"),
                              (send_telegram, "telegram")] if fn(message)]
    return sent
