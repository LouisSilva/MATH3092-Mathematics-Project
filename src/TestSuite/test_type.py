from enum import Enum

class TestType(Enum):
    POSITIVE = "Positive"  # Query is in the DB
    NEGATIVE = "Negative"  # Query is NOT in the DB (Alien)

    def __str__(self):
        return self.value