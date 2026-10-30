import socket

import pytest


def test_network_is_blocked():
    with pytest.raises(RuntimeError, match="network access is blocked"):
        socket.create_connection(("example.com", 80))
    with pytest.raises(RuntimeError, match="network access is blocked"):
        socket.getaddrinfo("example.com", 80)


def test_cv_package_imports_without_heavy_dependencies():
    import subprocess
    import sys

    code = ("import sys, agentic.cv; "
            "bad = [m for m in ('torch', 'transformers', 'tensorflow', 'ultralytics') if m in sys.modules]; "
            "sys.exit(1 if bad else 0)")
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0
