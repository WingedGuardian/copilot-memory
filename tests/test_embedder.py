"""Basic import and structure tests for copilot_memory."""
import pytest

def test_imports():
    from copilot_memory import Embedder, MemoryManager, Episode, EpisodicStore, FullTextStore, FTSResult

def test_embedder_init():
    from copilot_memory import Embedder
    e = Embedder(api_base="http://localhost:1234/v1")
    assert e._dimensions == 768

def test_episode_dataclass():
    from copilot_memory import Episode
    ep = Episode(id="test", text="hello world")
    assert ep.text == "hello world"
    assert ep.score == 0.0
