"""Check the current policy with synthetic cache events, without using traces or attempts."""
from local_client import request

raise SystemExit(request({'action':'smoke_test'}))
