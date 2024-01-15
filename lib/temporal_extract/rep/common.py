from pydantic import BaseModel


class Instance(BaseModel):
    """An instance of something in some document has a unique ID.
    The ID is unique within that document."""

    id_: str


class Span(Instance):
    """A span is an instance with start and end position in some text/document"""

    start: int
    end: int
