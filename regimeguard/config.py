"""Central configuration — override via environment variables."""
import os

TELEGRAM_BOT_TOKEN = os.getenv("RG_TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID   = os.getenv("RG_TELEGRAM_CHAT_ID", "")
DISCORD_WEBHOOK    = os.getenv("RG_DISCORD_WEBHOOK", "")

TRADING_DAYS = 252
CACHE_DIR    = os.getenv("RG_CACHE_DIR", ".cache")
MAX_POSITION_WEIGHT = 0.25
PORTFOLIO_VOL_TARGET = 0.15

STRESS_EVENTS = {
    "2008 GFC":         ("2008-09-15", "2009-03-09"),
    "COVID Crash":      ("2020-02-19", "2020-03-23"),
    "2022 Bear":        ("2022-01-04", "2022-10-13"),
    "2018 Volmageddon": ("2018-02-01", "2018-02-08"),
}
