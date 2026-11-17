from pathlib import Path

import pytest

from spidy.brain import study as st


def _session_with(pages):
    s = st.StudySession()
    s.doc = st.StudyDoc(name="Cloud Computing Notes.pdf", path="x.pdf", pages=pages)
    return s


def test_relevant_pages_and_named_page_go_with_the_question():
    pages = ["Intro to the course"] + ["filler text about networking"] * 5 + \
            ["Cloud computing is the on-demand delivery of computing services over the internet."] + \
            ["Grid computing connects many computers"]
    s = _session_with(pages)
    ctx = s.context_for("explain the definition of cloud computing")
    assert "[Page 7]" in ctx and "on-demand delivery" in ctx and "STUDY MODE" in ctx
    ctx = s.context_for("what is on page 8")
    assert "[Page 8]" in ctx and s.current_page == 8


def test_screen_page_counter_sets_the_current_page():
    s = _session_with(["a"] * 26)
    s.note_screen_text("Cloud Computing JNTUH ... 3 / 26  100%")
    assert s.current_page == 3


def test_questions_vs_commands_while_studying():
    s = _session_with(["text"])
    assert s.is_question("explain the definition of it")
    assert s.is_question("what are the types of cloud?")
    assert not s.is_question("open youtube")
    assert not st.StudySession().is_question("explain this")       # nothing loaded


def test_finds_the_open_pdf_file_by_name(tmp_path):
    d = tmp_path / "Downloads" / "sem5"
    d.mkdir(parents=True)
    (d / "Cloud Computing Notes.pdf").write_bytes(b"%PDF-1.4")
    assert st.pdf_name_from_title("Cloud Computing Notes.pdf - Google Chrome") == "Cloud Computing Notes.pdf"
    found = st.find_pdf_file("cloud computing notes.pdf", roots=[tmp_path / "Downloads"])
    assert found and Path(found).name == "Cloud Computing Notes.pdf"


def test_close_command_forgets_the_document():
    s = _session_with(["text"])
    assert "put" in s.try_command("close the pdf")
    assert not s.active
