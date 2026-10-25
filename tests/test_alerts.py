"""Alert transport checks with no external messages sent."""
from unittest.mock import Mock

import pytest
import requests

from regimeguard import alerts


@pytest.fixture(autouse=True)
def configured(monkeypatch):
    monkeypatch.setattr(alerts.config, 'DISCORD_WEBHOOK', 'https://example.invalid/hook')
    monkeypatch.setattr(alerts.config, 'TELEGRAM_BOT_TOKEN', 'test-token')
    monkeypatch.setattr(alerts.config, 'TELEGRAM_CHAT_ID', 'test-chat')


@pytest.mark.parametrize('send', [alerts.send_discord, alerts.send_telegram])
def test_http_failure_is_not_reported_as_sent(monkeypatch, send):
    response = Mock()
    response.raise_for_status.side_effect = requests.HTTPError('delivery rejected')
    monkeypatch.setattr(alerts.requests, 'post', Mock(return_value=response))
    with pytest.raises(alerts.AlertDeliveryError):
        send('test report')


@pytest.mark.parametrize('send', [alerts.send_discord, alerts.send_telegram])
def test_success_checks_response(monkeypatch, send):
    response = Mock()
    post = Mock(return_value=response)
    monkeypatch.setattr(alerts.requests, 'post', post)
    assert send('test report') is True
    response.raise_for_status.assert_called_once_with()
    assert post.call_args.kwargs['timeout'] == 10


@pytest.mark.parametrize('send', [alerts.send_discord, alerts.send_telegram])
def test_timeouts_propagate(monkeypatch, send):
    monkeypatch.setattr(alerts.requests, 'post', Mock(side_effect=requests.Timeout))
    with pytest.raises(alerts.AlertDeliveryError):
        send('test report')


def test_unconfigured_channels_make_no_requests(monkeypatch):
    monkeypatch.setattr(alerts.config, 'DISCORD_WEBHOOK', '')
    monkeypatch.setattr(alerts.config, 'TELEGRAM_BOT_TOKEN', '')
    post = Mock()
    monkeypatch.setattr(alerts.requests, 'post', post)
    assert alerts.notify_all('test report') == []
    post.assert_not_called()


def test_notify_all_lists_successful_channels(monkeypatch):
    monkeypatch.setattr(alerts.requests, 'post', Mock(return_value=Mock()))
    assert alerts.notify_all('test report') == ['discord', 'telegram']


def test_notify_all_uses_remaining_channel_after_failure(monkeypatch):
    discord = Mock(side_effect=alerts.AlertDeliveryError('Discord failed.'))
    telegram = Mock(return_value=True)
    monkeypatch.setattr(alerts, 'send_discord', discord)
    monkeypatch.setattr(alerts, 'send_telegram', telegram)

    assert alerts.notify_all('test report') == ['telegram']
    discord.assert_called_once_with('test report')
    telegram.assert_called_once_with('test report')


def test_notify_all_reports_when_every_configured_channel_fails(monkeypatch):
    monkeypatch.setattr(alerts, 'send_discord',
                        Mock(side_effect=alerts.AlertDeliveryError('Discord failed.')))
    monkeypatch.setattr(alerts, 'send_telegram',
                        Mock(side_effect=alerts.AlertDeliveryError('Telegram failed.')))

    with pytest.raises(alerts.AlertDeliveryError, match='All configured'):
        alerts.notify_all('test report')
