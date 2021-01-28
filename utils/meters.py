
class Meter:
    def __init__(self) -> None:
        self.value: float = 0
        self.count: int = 0

    def update(self, v, c=1) -> None:
        self.value += v
        self.count += c

    def reset(self) -> None:
        self.value = 0
        self.count = 0

    def avg(self) -> float:
        assert(self.count != 0)
        return self.value/self.count
