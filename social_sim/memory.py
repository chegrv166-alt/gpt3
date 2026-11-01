"""Ограниченная выборка опыта; коэффициенты — технические параметры модели."""
from social_sim.models import Memory, NPCState

MEMORY_LIMIT = 24
RECALL_LIMIT = 5
RECENCY_BONUS = 10.0
APPRAISAL_GAIN = 0.08
UTILITY_GAIN = 2.0


def rank(memory: Memory, tick: int) -> tuple[float, int, str]:
    return (memory.significance * memory.confidence + RECENCY_BONUS / (1 + max(0, tick-memory.tick)), memory.tick, memory.id)


def retain_memories(memories: list[Memory], tick: int, limit: int = MEMORY_LIMIT) -> list[Memory]:
    unique = {(m.event_id, m.target, m.topic): m for m in memories}
    return sorted(unique.values(), key=lambda m: rank(m, tick), reverse=True)[:limit]


def relevant_memories(state: NPCState, target: str, topic: str, tick: int) -> tuple[Memory, ...]:
    candidates = [m for m in state.memories if m.target == target and m.topic == topic and m.tick <= tick]
    return tuple(retain_memories(candidates, tick, RECALL_LIMIT))


def memory_signal(memories: tuple[Memory, ...]) -> float:
    return sum(m.significance / 100 * m.confidence * (1 if m.valence == 'negative' else -1) for m in memories)
