import pytest
from app import format_pipe_string

def test_format_pipe_string():
    assert format_pipe_string("MACRO: BULLISH | MICRO: BEARISH | STATUS: CONFLICT") == "  • MACRO: BULLISH\n  • MICRO: BEARISH\n  • STATUS: CONFLICT"
    assert format_pipe_string("THREAT: SUPPORT | DISTANCE: 10% | ACTION: TESTING") == "  • THREAT: SUPPORT\n  • DISTANCE: 10%\n  • ACTION: TESTING"
    assert format_pipe_string("") == ""
    assert format_pipe_string(" SINGLE_VALUE ") == "  • SINGLE_VALUE"
    assert format_pipe_string("VAL 1 | VAL 2 | ") == "  • VAL 1\n  • VAL 2"
