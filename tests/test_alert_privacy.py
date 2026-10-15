"""Failed requests must not expose credential-bearing URLs in tracebacks."""
import traceback
from unittest.mock import Mock

import pytest
import requests

from regimeguard import alerts


@pytest.mark.parametrize('channel', ['discord', 'telegram'])
@pytest.mark.parametrize('failure', ['http', 'timeout', 'connection'])
def test_delivery_errors_hide_secrets(monkeypatch, channel, failure):
    secret = 'synthetic-secret-for-test'
    monkeypatch.setattr(alerts.config, 'DISCORD_WEBHOOK', 'https://example.invalid/' + secret)
    monkeypatch.setattr(alerts.config, 'TELEGRAM_BOT_TOKEN', secret)
    monkeypatch.setattr(alerts.config, 'TELEGRAM_CHAT_ID', 'test-chat')
    error_type = {'http': requests.HTTPError, 'timeout': requests.Timeout,
                  'connection': requests.ConnectionError}[failure]
    error = error_type('Request failed at https://example.invalid/' + secret)
    response = Mock()
    post = Mock(return_value=response)
    if failure == 'http':
        response.raise_for_status.side_effect = error
    else:
        post.side_effect = error
    monkeypatch.setattr(alerts.requests, 'post', post)
    message = 'private report'
    with pytest.raises(alerts.AlertDeliveryError) as caught:
        getattr(alerts, 'send_' + channel)(message)
    rendered = ''.join(traceback.format_exception(caught.type, caught.value, caught.tb))
    assert secret not in rendered
    assert message not in rendered
    assert channel in str(caught.value).lower()
