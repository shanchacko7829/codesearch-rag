import tempfile
from pathlib import Path

FILES = {
    "pkg/__init__.py": "",
    "pkg/auth.py": '''"""Authentication helpers."""
import hashlib


def hash_password(password, salt):
    """Return the salted sha256 digest of a password as a hex string."""
    return hashlib.sha256((salt + password).encode()).hexdigest()


def verify_password(password, salt, expected):
    """Check whether a password matches the stored salted digest."""
    return hash_password(password, salt) == expected


class SessionStore:
    """Keeps logged in user sessions in memory."""

    def create_session(self, user_id):
        """Create a new session token for the given user and remember it."""
        token = "tok-%s" % user_id
        self.sessions = {token: user_id}
        return token

    def drop_session(self, token):
        """Forget a session token so the user is logged out of that session."""
        self.sessions.pop(token, None)
''',
    "pkg/files.py": '''import os


def read_lines(path):
    """Read a text file and return its lines without trailing newlines."""
    with open(path) as fh:
        return [ln.rstrip("\\n") for ln in fh]


def list_python_files(root):
    """Walk a directory tree and collect every python source file path."""
    out = []
    for base, _, names in os.walk(root):
        out += [os.path.join(base, n) for n in names if n.endswith(".py")]
    return out
''',
    "pkg/broken.py": "def oops(:\n    pass\n",
    "tests/test_x.py": "def test_x():\n    assert True\n",
}


def make_repo():
    d = tempfile.mkdtemp()
    for rel, body in FILES.items():
        p = Path(d) / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body)
    return d
