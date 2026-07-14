"""Tests for spidy.learning.workflows - WorkflowLearner."""
import pytest
from spidy.learning.workflows import WorkflowLearner, _extract_subsequences

@pytest.fixture
def learner():
    return WorkflowLearner(db_path=":memory:", min_observations=3, max_steps=5)

def test_extract_subsequences_basic():
    seqs = _extract_subsequences(["a", "b", "c"], 2, 5)
    assert ("a", "b") in seqs
    assert ("b", "c") in seqs
    assert ("a", "b", "c") in seqs

def test_extract_subsequences_min_len():
    seqs = _extract_subsequences(["a", "b"], 2, 5)
    assert ("a", "b") in seqs
    # No length-1 sequences
    assert ("a",) not in seqs

def test_extract_subsequences_max_len():
    seqs = _extract_subsequences(["a", "b", "c", "d"], 2, 2)
    for seq in seqs:
        assert len(seq) == 2

def test_extract_subsequences_too_short():
    seqs = _extract_subsequences(["a"], 2, 5)
    assert seqs == []

@pytest.mark.asyncio
async def test_initialize(learner):
    await learner.initialize()

@pytest.mark.asyncio
async def test_record_and_flush_no_workflow(learner):
    await learner.initialize()
    await learner.record_step("sess1", "open_file")
    result = await learner.flush_session("sess1")
    assert result == []  # only 1 step, no subsequences

@pytest.mark.asyncio
async def test_flush_empty_session(learner):
    await learner.initialize()
    result = await learner.flush_session("nonexistent_session")
    assert result == []

@pytest.mark.asyncio
async def test_list_workflows_empty(learner):
    await learner.initialize()
    assert await learner.list_workflows() == []

@pytest.mark.asyncio
async def test_count_empty(learner):
    await learner.initialize()
    assert await learner.count() == 0

@pytest.mark.asyncio
async def test_workflow_promoted_after_repetitions():
    learner = WorkflowLearner(db_path=":memory:", min_observations=2, max_steps=5)
    await learner.initialize()
    steps = ["open_file", "edit_code", "git_commit"]
    for i in range(3):
        for s in steps:
            await learner.record_step(f"sess{i}", s)
        await learner.flush_session(f"sess{i}")
    workflows = await learner.list_workflows()
    assert len(workflows) >= 1

@pytest.mark.asyncio
async def test_suggest_next_step_no_workflow(learner):
    await learner.initialize()
    result = await learner.suggest_next_step(["open_file"])
    assert result is None

@pytest.mark.asyncio
async def test_suggest_next_step_with_workflow():
    learner = WorkflowLearner(db_path=":memory:", min_observations=2, max_steps=5)
    await learner.initialize()
    steps = ["step_a", "step_b", "step_c"]
    for i in range(3):
        for s in steps:
            await learner.record_step(f"sess{i}", s)
        await learner.flush_session(f"sess{i}")
    result = await learner.suggest_next_step(["step_a", "step_b"])
    assert result == "step_c"

@pytest.mark.asyncio
async def test_record_step_bounds_buffer(learner):
    await learner.initialize()
    # Add many steps — should not raise
    for i in range(20):
        await learner.record_step("sess1", f"step_{i}")
    await learner.flush_session("sess1")