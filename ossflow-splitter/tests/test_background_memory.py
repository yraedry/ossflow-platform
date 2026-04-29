"""Tests for chapter_splitter.detection.background_memory.BackgroundMemory."""

import pytest

from chapter_splitter.config import Config
from chapter_splitter.detection.background_memory import BackgroundMemory


@pytest.fixture
def memory():
    """BackgroundMemory with default config."""
    return BackgroundMemory(Config())


@pytest.fixture
def memory_small_capacity():
    """BackgroundMemory with max_entries=3 so pruning triggers quickly."""
    cfg = Config(background_max_entries=3)
    return BackgroundMemory(cfg)


class TestLearn:
    """Test learning / adding entries."""

    def test_learn_new_entry(self, memory):
        memory.learn("Guard Retention Basics")
        assert memory.is_background("Guard Retention Basics")

    def test_learn_ignores_short_text(self, memory):
        memory.learn("Hi")
        assert not memory.is_background("Hi")

    def test_learn_ignores_empty_string(self, memory):
        memory.learn("")
        assert not memory._entries


class TestIsBackground:
    """Test background matching."""

    def test_empty_memory_returns_false(self, memory):
        assert not memory.is_background("anything")

    def test_exact_match(self, memory):
        memory.learn("Guard Passing Concepts")
        assert memory.is_background("Guard Passing Concepts")

    def test_case_insensitive(self, memory):
        memory.learn("guard passing concepts")
        assert memory.is_background("GUARD PASSING CONCEPTS")

    def test_similar_text_matches(self, memory):
        memory.learn("Guard Passing Concepts")
        # Very similar string should match (similarity > 0.45 threshold)
        assert memory.is_background("Guard Passing Concept")

    def test_different_text_does_not_match(self, memory):
        memory.learn("Guard Passing Concepts")
        assert not memory.is_background("Arm Drag System Fundamentals")


class TestDecay:
    """Test weight decay behavior."""

    def test_decay_reduces_weight(self, memory):
        memory.learn("Guard Passing Concepts")
        initial_weight = memory._entries[0].weight
        memory.decay()
        decayed_weight = memory._entries[0].weight
        assert decayed_weight < initial_weight

    def test_decay_uses_config_factor(self):
        cfg = Config(background_decay=0.5)
        mem = BackgroundMemory(cfg)
        mem.learn("Guard Passing Concepts")
        mem.decay()
        assert mem._entries[0].weight == pytest.approx(0.5, abs=0.01)

    def test_multiple_decays_compound(self, memory):
        memory.learn("Guard Passing Concepts")
        for _ in range(10):
            memory.decay()
        assert memory._entries[0].weight < 0.6  # 0.95^10 ~ 0.598


class TestGarbageCollection:
    """Test that entries with low weight are garbage-collected."""

    def test_gc_removes_low_weight_entries(self):
        cfg = Config(background_decay=0.01)  # aggressive decay
        mem = BackgroundMemory(cfg)
        mem.learn("Guard Passing Concepts")
        # After aggressive decay, weight = 1.0 * 0.01 = 0.01 < 0.1
        mem.decay()
        assert len(mem._entries) == 0

    def test_gc_keeps_high_weight_entries(self):
        cfg = Config(background_decay=0.5)
        mem = BackgroundMemory(cfg)
        mem.learn("Guard Passing Concepts")
        # Boost weight several times
        for _ in range(10):
            mem.learn("Guard Passing Concepts")
        # weight is now ~11.0, after decay = 5.5, still > 0.1
        mem.decay()
        assert len(mem._entries) == 1


class TestDefinitiveBackground:
    """Test that high-weight entries are definitive background."""

    def test_high_weight_is_definitive(self, memory):
        # Learn the same text many times to boost weight above 5.0
        for _ in range(6):
            memory.learn("Guard Passing Concepts")
        assert memory._entries[0].weight >= 6.0
        assert memory.is_background("Guard Passing Concepts")


class TestTimestamp:
    """Test that entries have timestamps."""

    def test_entry_has_last_seen(self, memory):
        memory.learn("Guard Passing Concepts")
        assert memory._entries[0].last_seen > 0

    def test_last_seen_updates_on_relearn(self, memory):
        memory.learn("Guard Passing Concepts")
        first_seen = memory._entries[0].last_seen
        memory.learn("Guard Passing Concepts")
        assert memory._entries[0].last_seen >= first_seen


class TestPruning:
    """Test that entries are pruned when capacity is exceeded."""

    def test_prune_removes_lowest_weight(self, memory_small_capacity):
        mem = memory_small_capacity
        mem.learn("Alpha Entry")
        mem.learn("Beta Entry")
        mem.learn("Gamma Entry")
        # All three fit within max_entries=3
        assert len(mem._entries) == 3

        # Adding a 4th should prune to 3, dropping the lowest weight
        mem.learn("Delta Entry")
        assert len(mem._entries) == 3


class TestBoost:
    """Test that re-observing a similar entry boosts its weight."""

    def test_boost_on_reobservation(self, memory):
        memory.learn("Guard Passing Concepts")
        initial_weight = memory._entries[0].weight
        # Re-learn the same text
        memory.learn("Guard Passing Concepts")
        boosted_weight = memory._entries[0].weight
        assert boosted_weight > initial_weight

    def test_boost_does_not_add_duplicate(self, memory):
        memory.learn("Guard Passing Concepts")
        memory.learn("Guard Passing Concepts")
        assert len(memory._entries) == 1


class TestEntriesAsTuples:
    """Test legacy compatibility property."""

    def test_entries_as_tuples(self, memory):
        memory.learn("Guard Passing Concepts")
        tuples = memory.entries_as_tuples
        assert len(tuples) == 1
        assert tuples[0][0] == "GUARD PASSING CONCEPTS"
        assert tuples[0][1] == 1.0
