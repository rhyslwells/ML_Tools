from utils import log_message

def test_log_message(caplog):
    """Ensure logging outputs correct message."""
    with caplog.at_level("INFO"):
        log_message("Test log")
        assert "Test log" in caplog.text
