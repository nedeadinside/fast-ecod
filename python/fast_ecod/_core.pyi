# Stub file
from collections.abc import Sequence
from typing import final

from typing_extensions import Self

@final
class ECODScoreMethod:
    RIGHT: ECODScoreMethod
    LEFT: ECODScoreMethod
    AUTO: ECODScoreMethod
    MAX: ECODScoreMethod

    def __eq__(self, other: object) -> bool: ...
    def __int__(self) -> int: ...

@final
class InductiveECOD:
    def __init__(self): ...
    def fit(self, x: Sequence[Sequence[float]]) -> Self: ...
    def decision_function(
        self,
        x: Sequence[Sequence[float]],
        method: ECODScoreMethod = ECODScoreMethod.MAX,
    ) -> Sequence[float]: ...
