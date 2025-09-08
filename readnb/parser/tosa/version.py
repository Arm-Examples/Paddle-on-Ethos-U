class Version:
    major :int = 0
    minor :int = 0
    patch :int = 0

    def __init__(self, ver: str):
        parts = ver.split('.')
        parts += ['0'] * (3 - len(parts))
        self.major, self.minor, self.patch = map(int, parts[:3])

    def __str__(self):
        return f"{self.major}.{self.minor}.{self.patch}"

    def __lt__(self, other):
        return (self.major, self.minor, self.patch) < (other.major, other.minor, other.patch)

    def __eq__(self, other):
        return (self.major, self.minor, self.patch) == (other.major, other.minor, other.patch)

    def __gt__(self, other):
        return (self.major, self.minor, self.patch) > (other.major, other.minor, other.patch)

    def __le__(self, other):
        return (self.major, self.minor, self.patch) <= (other.major, other.minor, other.patch)

    def __ge__(self, other):
        return (self.major, self.minor, self.patch) >= (other.major, other.minor, other.patch)

    def __ne__(self, other):
        return (self.major, self.minor, self.patch) != (other.major, other.minor, other.patch)
