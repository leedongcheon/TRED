"""Dates as day ordinals, granularities, and date literals of the temporal operations.

A span is ``(start, end)``: inclusive day ordinals (``datetime.date.toordinal``).
"""

import calendar
import datetime as dt
import re
from typing import Iterable, List, Tuple

GRANS = ("day", "month", "year")
Span = Tuple[int, int]

# Years may be unpadded (TimelineCronQR has "731-01-01").
_POINT_RE = re.compile(r"^(\d{1,4})(?:-(\d{1,2}))?(?:-(\d{1,2}))?$")


def check_gran(gran: str) -> str:
    if gran not in GRANS:
        raise ValueError(f"unknown granularity {gran!r}; expected one of {GRANS}")
    return gran


def to_ordinal(year: int, month: int = 1, day: int = 1) -> int:
    return dt.date(year, month, day).toordinal()


def parse_kg_date(text: str) -> int:
    """Parse a full ``Y-MM-DD`` KG date into a day ordinal (ValueError if invalid)."""
    match = _POINT_RE.match(text.strip())
    if match is None or match.group(2) is None or match.group(3) is None:
        raise ValueError(f"not a full date: {text!r}")
    return to_ordinal(int(match.group(1)), int(match.group(2)), int(match.group(3)))


def _parse_point(text: str) -> Span:
    match = _POINT_RE.match(text.strip())
    if match is None:
        raise ValueError(f"bad date literal {text!r}; use YYYY, YYYY-MM or YYYY-MM-DD")
    year = int(match.group(1))
    if match.group(2) is None:
        return to_ordinal(year, 1, 1), to_ordinal(year, 12, 31)
    month = int(match.group(2))
    if match.group(3) is None:
        return to_ordinal(year, month, 1), to_ordinal(year, month, calendar.monthrange(year, month)[1])
    day = to_ordinal(year, month, int(match.group(3)))
    return day, day


def parse_date_literal(text: str) -> Span:
    """``YYYY``, ``YYYY-MM``, ``YYYY-MM-DD`` or ``A..B`` -> inclusive span (ValueError if invalid)."""
    text = text.strip()
    if ".." in text:
        left, right = text.split("..", 1)
        start, end = _parse_point(left)[0], _parse_point(right)[1]
        if start > end:
            raise ValueError(f"span {text!r} ends before it starts")
        return start, end
    return _parse_point(text)


def gran_key(ordinal: int, gran: str) -> int:
    """Integer key that compares dates at ``gran`` (day ordinal, month index, or year)."""
    if gran == "day":
        return int(ordinal)
    date = dt.date.fromordinal(int(ordinal))
    if gran == "month":
        return date.year * 12 + date.month - 1
    if gran == "year":
        return date.year
    raise ValueError(f"unknown granularity {gran!r}")


def format_date(ordinal: int, gran: str = "day") -> str:
    date = dt.date.fromordinal(int(ordinal))
    if gran == "day":
        return f"{date.year:04d}-{date.month:02d}-{date.day:02d}"
    if gran == "month":
        return f"{date.year:04d}-{date.month:02d}"
    if gran == "year":
        return f"{date.year:04d}"
    raise ValueError(f"unknown granularity {gran!r}")


def format_span(span: Span, gran: str = "day", collapse: bool = False) -> str:
    """``A..B`` at ``gran``; with ``collapse`` a span inside one ``gran`` unit prints as ``A``."""
    start, end = format_date(span[0], gran), format_date(span[1], gran)
    if collapse and start == end:
        return start
    return f"{start}..{end}"


def shift_ordinal(ordinal: int, n: int, unit: str) -> int:
    """Move a date by ``n`` days/months/years (month/year shifts clamp the day of month)."""
    if unit == "day":
        result = int(ordinal) + int(n)
        if not 1 <= result <= dt.date.max.toordinal():
            raise ValueError("shifted date is out of range")
        return result
    date = dt.date.fromordinal(int(ordinal))
    months = int(n) * (12 if unit == "year" else 1)
    if unit not in ("month", "year"):
        raise ValueError(f"unknown unit {unit!r}")
    index = date.year * 12 + date.month - 1 + months
    year, month = divmod(index, 12)
    month += 1
    if not 1 <= year <= 9999:
        raise ValueError("shifted date is out of range")
    day = min(date.day, calendar.monthrange(year, month)[1])
    return to_ordinal(year, month, day)


def merge_spans(spans: Iterable[Span], gran: str) -> List[Span]:
    """Merge spans that overlap or touch when compared at ``gran``."""
    merged: List[List[int]] = []
    for start, end in sorted(spans):
        if merged and gran_key(start, gran) <= gran_key(merged[-1][1], gran):
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]
