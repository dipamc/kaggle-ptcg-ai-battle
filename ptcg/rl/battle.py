"""Multi-battle engine wrapper.

cg.game funnels everything through the module-level `Battle` singleton
(one battle_ptr per process); the underlying C functions all take the
pointer explicitly, so this wrapper manages one pointer per handle and
many concurrent battles per process — required for a vectorized env.
"""
import ctypes
import json

from cg.sim import lib


class BattleHandle:
    __slots__ = ("ptr",)

    def __init__(self, deck0: list[int], deck1: list[int]):
        assert len(deck0) == 60 and len(deck1) == 60
        cards = (ctypes.c_int * 120)(*deck0, *deck1)
        sd = lib.BattleStart(cards)
        if not sd.battlePtr:
            raise RuntimeError(
                f"BattleStart failed: player={sd.errorPlayer} type={sd.errorType}")
        self.ptr = sd.battlePtr

    def obs(self) -> dict:
        sd = lib.GetBattleData(self.ptr)
        return json.loads(sd.json.decode())

    def select(self, indices: list[int]) -> dict:
        arg = (ctypes.c_int * len(indices))(*indices)
        err = lib.Select(self.ptr, arg, len(indices))
        if err != 0:
            raise RuntimeError(f"Select error {err} for {indices}")
        return self.obs()

    def visualize(self) -> dict:
        return json.loads(lib.VisualizeData(self.ptr).decode())

    def finish(self):
        if self.ptr:
            lib.BattleFinish(self.ptr)
            self.ptr = None
