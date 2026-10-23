#!/usr/bin/env python3
"""PreToolUse hook: block any tool call that touches the ICRN token directory.

Claude Code passes the pending tool call as JSON on stdin. Exit code 2 blocks
the call and shows stderr to the agent. remote.py reads the token inside its
own process, so it is unaffected.
"""

import json
import re
import sys

PATTERN = re.compile(r"\.config/+icrn|icrn/+token", re.IGNORECASE)


def strings(obj):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from strings(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from strings(v)


call = json.load(sys.stdin)
if any(PATTERN.search(s) for s in strings(call.get("tool_input", {}))):
    print(
        "Blocked: the ICRN token is off-limits. remote.py reads it itself; "
        "use `uv run remote.py ...` instead.",
        file=sys.stderr,
    )
    sys.exit(2)
