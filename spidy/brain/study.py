"""
Study mode: Spidy reads your PDF and tutors you on it
=====================================================
Like asking Gemini / ChatGPT / Claude about a document:

* "read this pdf", "look at my screen, I opened a pdf", "explain this pdf"
      -> finds the PDF open in your browser / viewer (by its window title), locates the file on
         disk (Downloads, Desktop, Documents, OneDrive) and reads every page.
* "attach pdf C:\\path\\notes.pdf"  (what the 📎 button in the chat sends)
      -> reads that file.
* every question after that ("explain the definition of cloud computing", "what's on page 5",
  "summarise unit 2") is answered from the document: the most relevant pages (plus the page you
  are on, or any page you name) go to the language model with the question.
* "close the pdf" / "stop studying" -> forget the document.

The document text is passed to the model per question and never stored in the conversation
history, so the chat stays small and your earlier conversation is untouched.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from spidy.logging.logger import get_logger

log = get_logger(__name__)

MAX_CONTEXT_CHARS = 14_000          # material sent with one question
CHUNK_CHARS = 1_400

_STOPWORDS = set("""a an the of to in on for and or is are was were be been it this that these those
what which who whom how why when where can could would should do does did you your me my i we our
please explain tell about give define definition meaning page pdf document notes spidy hey ok okay
with from by as at into than then so if not no yes also just more some any all""".split())

_READ_PDF = re.compile(
    r"\b(read|open|load|study|look\s+(?:at|into)|check|see|explain|summari[sz]e|teach\s+me)\b.*\b(pdf|document|notes|file)\b"
    r"|\b(opened|have)\s+(?:a|the|my)\s+pdf\b|\bthis\s+pdf\b|\bmy\s+pdf\b|\bthe\s+pdf\b",
    re.I)
_ATTACH = re.compile(r"^\s*attach\s+pdf\s+(.+?)\s*$", re.I)
_CLOSE = re.compile(r"\b(close|forget|stop\s+studying|done\s+studying|remove)\b.*\b(pdf|document|notes|studying)\b"
                    r"|\bstop\s+studying\b", re.I)
_PAGE = re.compile(r"\bpage\s*(?:no\.?|number)?\s*(\d{1,4})\b", re.I)
_SEARCH_DIRS = ("Downloads", "Desktop", "Documents", "OneDrive")


@dataclass
class StudyDoc:
    name: str
    path: str
    pages: list[str] = field(default_factory=list)

    @property
    def n_pages(self) -> int:
        return len(self.pages)


def extract_pdf_text(path: str) -> list[str]:
    """Text of every page (pypdf). Raises RuntimeError with a friendly reason if it can't."""
    try:
        import pypdf
    except ImportError as exc:
        raise RuntimeError("the PDF reader (pypdf) isn't installed yet") from exc
    reader = pypdf.PdfReader(path)
    pages = []
    for pg in reader.pages:
        try:
            pages.append(re.sub(r"[ \t]+", " ", pg.extract_text() or "").strip())
        except Exception:  # noqa: BLE001 -- one bad page must not lose the whole document
            pages.append("")
    return pages


def open_pdf_titles() -> list[str]:
    """Titles of open windows that show a PDF (browser tab / PDF viewer), foreground first."""
    titles: list[str] = []
    try:
        import win32gui
        fg = win32gui.GetForegroundWindow()

        def add(h, _):
            if win32gui.IsWindowVisible(h):
                t = win32gui.GetWindowText(h)
                if ".pdf" in t.lower():
                    titles.append((h == fg, t))
        win32gui.EnumWindows(add, None)
        titles.sort(key=lambda x: not x[0])
        return [t for _, t in titles]
    except Exception:  # noqa: BLE001
        return []


def pdf_name_from_title(title: str) -> str | None:
    m = re.search(r"([^\\/:*?\"<>|]+?\.pdf)\b", title, re.I)
    return m.group(1).strip() if m else None


def find_pdf_file(name: str, roots: list[Path] | None = None, max_depth: int = 4) -> str | None:
    """Look for a PDF with this file name in the usual places (Downloads first)."""
    home = Path.home()
    roots = roots or [home / d for d in _SEARCH_DIRS]
    want = name.lower()
    for root in roots:
        if not root.is_dir():
            continue
        base = len(root.parts)
        for dirpath, dirnames, filenames in os.walk(root):
            if len(Path(dirpath).parts) - base >= max_depth:
                dirnames[:] = []
            dirnames[:] = [d for d in dirnames if not d.startswith((".", "node_modules", "__pycache__"))]
            for f in filenames:
                if f.lower() == want:
                    return str(Path(dirpath) / f)
    return None


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 2 and w not in _STOPWORDS}


class StudySession:
    def __init__(self) -> None:
        self.doc: StudyDoc | None = None
        self.current_page: int | None = None       # 1-based, when known (named or seen on screen)

    @property
    def active(self) -> bool:
        return self.doc is not None and any(self.doc.pages)

    # ── loading ─────────────────────────────────────────────────────────────
    def load(self, path: str) -> str:
        try:
            pages = extract_pdf_text(path)
        except RuntimeError as exc:
            return f"I found the PDF, but I can't read it: {exc}."
        except Exception as exc:  # noqa: BLE001
            log.warning("[STUDY] could not read {p}: {e}", p=path, e=exc)
            return "I found the PDF, but couldn't read it. It may be scanned images or protected."
        name = Path(path).name
        if not any(pages):
            return f"'{name}' has no text I can read: it's probably scanned images."
        self.doc, self.current_page = StudyDoc(name=name, path=path, pages=pages), None
        log.info("[STUDY] loaded '{n}' ({k} pages)", n=name, k=len(pages))
        return (f"Got it! I've read '{Path(name).stem}', all {len(pages)} pages. "
                f"Ask me anything about it: definitions, explanations, summaries or exam answers.")

    def load_open_pdf(self) -> str | None:
        """Read the PDF that is open on screen. None if no PDF window is open."""
        titles = open_pdf_titles()
        if not titles:
            return None
        for title in titles:
            name = pdf_name_from_title(title)
            if not name:
                continue
            if self.doc and self.doc.name.lower() == name.lower():
                return f"I already have '{Path(name).stem}' open, all {self.doc.n_pages} pages. Ask away!"
            path = find_pdf_file(name)
            if path:
                return self.load(path)
            return (f"I can see '{Path(name).stem}' on your screen, but it's open from the internet, so "
                    f"there's no file on this PC for me to read. Press Ctrl+S in the browser to save it "
                    f"to Downloads, then say 'read this PDF' again, or attach it from my 📎 menu.")
        return None

    # ── commands ────────────────────────────────────────────────────────────
    def try_command(self, utterance: str) -> str | None:
        """Handle study commands; None means "not a study command"."""
        m = _ATTACH.match(utterance)
        if m:
            path = m.group(1).strip().strip('"')
            if not os.path.isfile(path):
                return "I couldn't find that file."
            return self.load(path)
        if _CLOSE.search(utterance) and self.doc is not None:
            name = Path(self.doc.name).stem
            self.doc, self.current_page = None, None
            return f"Okay, I've put '{name}' away."
        if _READ_PDF.search(utterance) and not self._is_question_about_loaded(utterance):
            reply = self.load_open_pdf()
            if reply is not None:
                return reply
            if self.doc is None:
                return ("I don't see a PDF open right now. Open it in your browser, or attach it with "
                        "the 📎 button in my chat.")
        return None

    def _is_question_about_loaded(self, utterance: str) -> bool:
        """With a document already loaded, "explain this pdf's unit 2" is a question, not a reload."""
        if self.doc is None:
            return False
        titles = open_pdf_titles()
        names = {(pdf_name_from_title(t) or "").lower() for t in titles}
        same_doc_open = not titles or self.doc.name.lower() in names
        asks = bool(re.search(r"\b(explain|what|why|how|define|summari[sz]e|tell|list|compare|difference)\b",
                              utterance, re.I))
        return same_doc_open and asks

    # ── answering ───────────────────────────────────────────────────────────
    def is_question(self, utterance: str) -> bool:
        """While studying, does this look like a question for the tutor (not a command like "open youtube")?"""
        if not self.active:
            return False
        u = utterance.strip().lower()
        if re.match(r"^(open|close|play|search|set|start|launch|go to|take|turn|volume|mute|lock|remind)\b", u):
            return False
        return bool(re.search(r"\b(explain|what|why|how|define|definition|meaning|summari[sz]e|tell me|list|"
                              r"compare|difference|example|page|unit|chapter|answer|notes?|mark|describe|"
                              r"elaborate|simplify|quiz|question)\b", u)) or u.endswith("?")

    def note_screen_text(self, text: str) -> None:
        """Pick up "3 / 26" or "Page 3 of 26" from what's on screen: the page you are reading."""
        if not self.doc or not text:
            return
        for m in re.finditer(r"\b(\d{1,4})\s*(?:/|of)\s*(\d{1,4})\b", text):
            cur, total = int(m.group(1)), int(m.group(2))
            if total == self.doc.n_pages and 1 <= cur <= total:
                self.current_page = cur
                return

    def context_for(self, question: str) -> str | None:
        """The study material to send with this question (None when no document is loaded)."""
        if not self.active:
            return None
        doc = self.doc
        named = [int(n) for n in _PAGE.findall(question) if 1 <= int(n) <= doc.n_pages]
        if named:
            self.current_page = named[0]
        q = _words(question)
        chunks: list[tuple[float, int, str]] = []
        for i, page in enumerate(doc.pages):
            for j in range(0, max(1, len(page)), CHUNK_CHARS):
                part = page[j:j + CHUNK_CHARS]
                if not part.strip():
                    continue
                w = _words(part)
                score = len(q & w) / (1 + len(q)) if q else 0.0
                if i + 1 in named:
                    score += 10
                elif self.current_page and abs(i + 1 - self.current_page) <= 1:
                    score += 0.6 if i + 1 == self.current_page else 0.25
                chunks.append((score, i, part))
        chunks.sort(key=lambda c: -c[0])
        picked, used = [], 0
        for score, i, part in chunks:
            if used + len(part) > MAX_CONTEXT_CHARS:
                continue
            picked.append((i, part))
            used += len(part)
            if used > MAX_CONTEXT_CHARS * 0.9:
                break
        if not picked:                                   # vague question: start of the document
            picked = [(i, p[:CHUNK_CHARS]) for i, p in enumerate(doc.pages[:6]) if p]
        picked.sort()
        body = "\n\n".join(f"[Page {i + 1}]\n{part}" for i, part in picked)
        where = f" The user is currently on page {self.current_page}." if self.current_page else ""
        return (f"STUDY MODE. The user is studying the document '{Path(doc.name).stem}' "
                f"({doc.n_pages} pages).{where} Answer like a patient tutor, using the excerpts below "
                f"as the source: explain clearly with simple examples, keep exam-style answers "
                f"structured, and mention the page numbers you used. If the excerpts don't cover the "
                f"question, say so and answer from general knowledge.\n\n{body}")
