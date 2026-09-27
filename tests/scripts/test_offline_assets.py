import importlib.util
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("audit", ROOT / "benchmarks/tools/check_offline_assets.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class OfflineAuditTest(unittest.TestCase):
    def check(self, name, content):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / name).write_text(content)
            return audit.scan(root)[0]

    def test_allowlisted_host_is_not_allowed_for_active_resources(self):
        self.assertTrue(self.check("index.html", '<script src="https://github.com/a.js"></script>'))
        self.assertTrue(self.check("style.css", '@import "//github.com/a.css";'))
        self.assertTrue(self.check("app.js", 'fetch("https://github.com/api")'))

    def test_protocol_relative_resources(self):
        self.assertTrue(self.check("index.html", '<img src="//cdn.example.com/a.png">'))

    def test_attribution_and_local_resources(self):
        self.assertFalse(self.check("index.html", '<a href="https://github.com/project">Source</a><script src="app.js"></script>'))
        self.assertFalse(self.check("style.css", '@font-face{src:url(font.woff2)}'))


if __name__ == "__main__":
    unittest.main()
