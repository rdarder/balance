from typing import Optional, TypeVar

T = TypeVar("T")


class InvalidState(Exception):
    def __init__(self, message: str):
        self.message = message


class InvalidArgument(Exception):
    def __init__(self, message: str):
        self.message = message


class UnexpectedNoneValue(Exception):
    def __init__(self, message: str):
        self.message = message


def check_state(condition: bool, message: str) -> None:
    if not condition:
        raise InvalidState(message)


def check_argument(condition: bool, message: str) -> None:
    if not condition:
        raise InvalidArgument(message)


def check_not_none(value: Optional[T], message: Optional[str] = None) -> T:
    if value is None:
        raise UnexpectedNoneValue(message or "Value cannot be None")
    return value
