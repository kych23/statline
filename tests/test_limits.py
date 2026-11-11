import pytest

from statline.server.limits import (
    PUBLIC_CALLER,
    Busy,
    HeavyToolGate,
    OverQuota,
    caller_id,
    quota_caller,
)


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


async def use(gate: HeavyToolGate, caller: str = "a") -> None:
    async with gate.slot(caller):
        pass


@pytest.mark.anyio
async def test_quota_exhausts_per_caller_then_resets() -> None:
    clock = FakeClock()
    gate = HeavyToolGate(2, clock)
    await use(gate)
    clock.now += 100
    await use(gate)
    with pytest.raises(OverQuota) as exc:
        await use(gate)
    assert exc.value.resets_in_s == 3500
    await use(gate, "b")
    clock.now += 3500
    await use(gate)


@pytest.mark.anyio
async def test_busy_while_slot_held_and_not_counted() -> None:
    gate = HeavyToolGate(5, FakeClock())
    async with gate.slot("a"):
        with pytest.raises(Busy):
            await use(gate, "b")
    await use(gate, "b")
    assert len(gate._calls["b"]) == 1


@pytest.mark.anyio
async def test_history_is_pruned_after_window() -> None:
    clock = FakeClock()
    gate = HeavyToolGate(5, clock)
    await use(gate, "a")
    await use(gate, "b")
    clock.now += 3600
    await use(gate, "c")
    assert set(gate._calls) == {"c"}


@pytest.mark.parametrize(
    ("header", "expected"),
    [
        ("1.1.1.1, 2.2.2.2", "ip:2.2.2.2"),
        ("  3.3.3.3  ", "ip:3.3.3.3"),
        ("1.1.1.1, ", "ip:1.1.1.1"),
        ("", "anon"),
        (" , ", "anon"),
        (None, "anon"),
    ],
)
def test_caller_id(header: str | None, expected: str) -> None:
    assert caller_id(header) == expected


@pytest.mark.anyio
async def test_public_bucket_has_its_own_limit() -> None:
    gate = HeavyToolGate(1, FakeClock(), public_per_hour=3)
    for _ in range(3):
        await use(gate, PUBLIC_CALLER)
    with pytest.raises(OverQuota):
        await use(gate, PUBLIC_CALLER)
    await use(gate, quota_caller("guid-a"))
    with pytest.raises(OverQuota):
        await use(gate, quota_caller("guid-a"))
    await use(gate, quota_caller("guid-b"))


def test_quota_caller() -> None:
    assert quota_caller("ABC") == "yahoo:ABC"
    assert quota_caller(None) == PUBLIC_CALLER
    assert quota_caller("") == PUBLIC_CALLER
