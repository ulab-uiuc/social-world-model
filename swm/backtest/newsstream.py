"""The global jin10 news stream, reconstructed from the per-record news lists.

Each dataset record ships the ~50 jin10 headlines published in the hour or so
before its move. Those windows overlap heavily across markets, so the union of
every record's list is the jin10 wire itself: 114k de-duplicated English
headlines with publication timestamps.

Working from the union rather than a record's own list matters for the
backtest. A record's list was assembled around that market's move; at a given
timestamp the union is strictly larger (median 73 vs 51 headlines per window),
so retrieval has to actually pick the relevant items out of the full wire
instead of choosing from a pre-narrowed shortlist.
"""

import bisect
import datetime as dt
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

_NORMALISE = re.compile(r'[^a-z0-9]+')


def parse_published_at(value: str | None) -> int | None:
    if not value:
        return None
    try:
        stamp = dt.datetime.strptime(value, '%Y-%m-%dT%H:%M:%SZ')
    except ValueError:
        try:
            stamp = dt.datetime.fromisoformat(value.replace('Z', '+00:00'))
        except ValueError:
            return None
    return int(stamp.replace(tzinfo=dt.timezone.utc).timestamp())


@dataclass
class NewsItem:
    t: int
    title: str
    description: str
    important: int
    raw: dict

    @property
    def text(self) -> str:
        body = self.description or ''
        if body and body.strip() == (self.title or '').strip():
            body = ''
        return f'{self.title}\n{body}'.strip()

    def as_prompt_dict(self) -> dict:
        return {
            'title': self.title,
            'description': self.description,
            'published_at': self.raw.get('published_at'),
            'source': self.raw.get('source', 'jin10'),
            'important': self.important,
        }


class NewsStream:
    """De-duplicated, time-sorted jin10 headlines with window slicing."""

    def __init__(self, records: Iterable[dict]):
        seen: dict[tuple, NewsItem] = {}
        for record in records:
            for raw in record.get('news') or []:
                t = parse_published_at(raw.get('published_at'))
                if t is None:
                    continue
                key = ((raw.get('title') or '').strip(), t)
                if key in seen:
                    continue
                seen[key] = NewsItem(
                    t=t,
                    title=(raw.get('title') or '').strip(),
                    description=(raw.get('description') or '').strip(),
                    important=int(raw.get('important') or 0),
                    raw=raw,
                )
        self.items: list[NewsItem] = sorted(seen.values(), key=lambda x: x.t)
        self._times = [item.t for item in self.items]

    def __len__(self) -> int:
        return len(self.items)

    def window(self, start_t: int, end_t: int) -> list[NewsItem]:
        """Headlines published in [start_t, end_t)."""
        lo = bisect.bisect_left(self._times, start_t)
        hi = bisect.bisect_left(self._times, end_t)
        return self.items[lo:hi]

    def index_range(self, start_t: int, end_t: int) -> range:
        return range(
            bisect.bisect_left(self._times, start_t),
            bisect.bisect_left(self._times, end_t),
        )

    @staticmethod
    def dedupe(items: Sequence['NewsItem']) -> list['NewsItem']:
        """Collapse re-publications of the same headline within a window.

        The wire repeats a story under a lightly reworded headline minutes
        apart. Left in, the duplicates crowd out the rest of the top-k and pile
        routing mass onto one story; the earliest copy is the one a live system
        would have acted on, so that is the one kept.
        """
        seen, out = set(), []
        for item in sorted(items, key=lambda x: x.t):
            key = _NORMALISE.sub('', (item.title or '').lower())[:80]
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(item)
        return out

    def distinct_windows(
        self, anchors: Sequence[int], lookback: int, lead: int
    ) -> dict[int, list[NewsItem]]:
        """Map each anchor timestamp to its [anchor-lookback, anchor-lead) slice."""
        return {a: self.window(a - lookback, a - lead) for a in anchors}
