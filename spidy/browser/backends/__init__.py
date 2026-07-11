"""
Browser Backends Package
"""
from spidy.browser.backends.base import BrowserBackend
from spidy.browser.backends.playwright_backend import PlaywrightBackend

__all__ = ["BrowserBackend", "PlaywrightBackend"]
