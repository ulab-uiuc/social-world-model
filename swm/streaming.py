"""Streaming contexts: a market's own past, packed into the context window.

The shipped world model reads one news headline at a time. Each of its prompts
carries a market's question, sixteen daily prices and a single headline, in
1024 tokens, and the per-headline predictions are averaged by attribution
weight. Nothing in that prompt says how this market has reacted to news before.

A streaming context says exactly that. It walks the market's decision points in
time order and, for each one already in the past, shows the headlines that
landed and the price change that followed:

    [2026-06-02 14:00]  price 0.420
      - Fed holds rates steady; BTC ETF inflows hit $1.2B
      - SEC delays decision on staking
      => 24h change: +0.055  (0.420 -> 0.475)

    [2026-06-05 09:00]  price 0.510
      - CPI comes in hot; risk assets sell off
      => 24h change: -0.083  (0.510 -> 0.427)

    [2026-06-17 20:00]  price 0.635
      - <the current window, every headline in it>
      => 24h change:

That shape serves two experiments off one builder. Fine-tuned, the model can
learn how a market's own reaction history conditions the next move. Untouched,
an instruct model can read the same demonstrations and answer in-context, which
is the baseline the fine-tune has to beat.

Two things the builder guarantees, because both experiments are worthless
without them:

  * only decision points strictly earlier than the target are ever shown, and
    their realised changes were knowable a day after they occurred, which is
    still before the target;
  * blocks are packed newest-first against a token budget, so a context is
    truncated at the far (oldest) end rather than losing the current headlines.
"""

import datetime as dt
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

DAY = 86400


def _stamp(t: int) -> str:
    return dt.datetime.fromtimestamp(int(t), dt.timezone.utc).strftime('%Y-%m-%d %H:%M')


def _date(t: int) -> str:
    return dt.datetime.fromtimestamp(int(t), dt.timezone.utc).strftime('%Y-%m-%d')


def record_time(record: dict[str, Any]) -> int:
    return int((record.get('target') or {}).get('t') or record.get('move_hour_t') or 0)


def anchor_price(record: dict[str, Any]) -> float | None:
    """The 24h-ago price a change is measured from -- the model's anchor."""
    history = record.get('history') or []
    return float(history[-1]['p']) if history else None


def realised_change(record: dict[str, Any]) -> float | None:
    before = anchor_price(record)
    target = (record.get('target') or {}).get('p')
    if before is None or target is None:
        return None
    return float(target) - before


@dataclass
class ContextStats:
    """What actually made it into a context, for the ablation to report."""

    own_blocks: int = 0
    sibling_blocks: int = 0
    current_news: int = 0
    dropped_blocks: int = 0
    approx_tokens: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            'own_blocks': self.own_blocks,
            'sibling_blocks': self.sibling_blocks,
            'current_news': self.current_news,
            'dropped_blocks': self.dropped_blocks,
            'approx_tokens': self.approx_tokens,
        }


@dataclass
class StreamingContextBuilder:
    """Builds one long prompt per record from that record's own past.

    `max_news_per_block` caps how many headlines a *past* decision point
    contributes; the current window is capped separately by
    `max_current_news` and is packed first, so a long context never crowds out
    the headlines the question is actually about.

    Sibling blocks (other markets under the same `event_id`) fill in when a
    market's own stream is thin, which it usually is: the median market has two
    prior decision points and 31% have none, against a median of five and 18%
    empty at the event level. They are labelled in the text so the model is not
    told a sibling's history is this market's own.
    """

    max_tokens: int = 16384
    max_news_per_block: int = 6
    max_current_news: int = 60
    max_blocks: int = 40
    chars_per_token: float = 3.6  # conservative for English news text
    include_siblings: bool = True
    max_description_chars: int = 300

    def _fits(self, text: str, used: int) -> bool:
        return used + len(text) / self.chars_per_token <= self.max_tokens

    def _news_lines(self, news: Sequence[dict[str, Any]], limit: int) -> list[str]:
        lines = []
        for item in list(news)[:limit]:
            title = (item.get('title') or '').strip()
            if not title:
                continue
            lines.append(f'  - {title}')
        return lines

    def _past_block(self, record: dict[str, Any], own: bool) -> str | None:
        change = realised_change(record)
        before = anchor_price(record)
        if change is None or before is None:
            return None
        header = f'[{_stamp(record_time(record))}]  price {before:.3f}'
        if not own:
            header += f'  (sibling market: {record.get("question", "")[:80]})'
        lines = [header]
        lines.extend(self._news_lines(record.get('news') or [], self.max_news_per_block))
        lines.append(
            f'  => 24h change: {change:+.3f}  ({before:.3f} -> {before + change:.3f})'
        )
        return '\n'.join(lines)

    def _current_block(self, record: dict[str, Any], stats: ContextStats) -> str:
        before = anchor_price(record)
        news = list(record.get('news') or [])[: self.max_current_news]
        stats.current_news = len(news)
        lines = [f'[{_stamp(record_time(record))}]  price {before:.3f}   <-- predict this one']
        lines.extend(self._news_lines(news, self.max_current_news))
        lines.append('  => 24h change:')
        return '\n'.join(lines)

    def _header(self, record: dict[str, Any]) -> str:
        lines = [f'Market: {record.get("question", "")}']
        description = (record.get('description') or '').strip()
        if description:
            lines.append(f'Description: {description[: self.max_description_chars]}')
        history = record.get('history') or []
        if history:
            lines.append('\nDaily price history:')
            lines.extend(f'  {_date(p["t"])}: {p["p"]:.3f}' for p in history)
        lines.append(
            '\nHow this market has moved after news, most recent last. '
            'Each block is one decision point: the price at the time, the '
            'headlines that landed, and the change over the following 24h.'
        )
        return '\n'.join(lines)

    def build(
        self,
        record: dict[str, Any],
        own_history: Sequence[dict[str, Any]],
        sibling_history: Sequence[dict[str, Any]] = (),
    ) -> dict[str, Any]:
        """Assemble the prompt. `*_history` must already be strictly earlier."""
        stats = ContextStats()
        header = self._header(record)
        current = self._current_block(record, stats)
        used = (len(header) + len(current)) / self.chars_per_token

        # Newest first so the budget is spent on the most recent past, then
        # reversed at the end to read forward in time.
        candidates = [(r, True) for r in sorted(own_history, key=record_time, reverse=True)]
        if self.include_siblings:
            candidates += [
                (r, False)
                for r in sorted(sibling_history, key=record_time, reverse=True)
            ]

        blocks: list[str] = []
        for past, own in candidates[: self.max_blocks]:
            block = self._past_block(past, own)
            if block is None:
                continue
            if not self._fits(block + '\n\n', used):
                stats.dropped_blocks += 1
                continue
            used += len(block + '\n\n') / self.chars_per_token
            blocks.append(block)
            if own:
                stats.own_blocks += 1
            else:
                stats.sibling_blocks += 1
        stats.dropped_blocks += max(0, len(candidates) - self.max_blocks)

        body = '\n\n'.join(reversed(blocks))
        prompt = f'{header}\n\n{body}\n\n{current}' if body else f'{header}\n\n{current}'
        stats.approx_tokens = int(len(prompt) / self.chars_per_token)
        return {'prompt': prompt, 'stats': stats.as_dict()}


def index_history(records: Iterable[dict[str, Any]]):
    """(by_market, by_event) -> time-sorted record lists, for slicing the past."""
    by_market: dict[str, list[dict]] = {}
    by_event: dict[str, list[dict]] = {}
    for record in records:
        by_market.setdefault(str(record.get('market_id')), []).append(record)
        by_event.setdefault(str(record.get('event_id')), []).append(record)
    for bucket in (by_market, by_event):
        for key in bucket:
            bucket[key].sort(key=record_time)
    return by_market, by_event


def past_of(
    record: dict[str, Any],
    by_market: dict[str, list[dict]],
    by_event: dict[str, list[dict]],
):
    """Strictly-earlier decision points: (own market, same-event siblings).

    The cutoff is `< target.t` on both sides. A block's realised change was
    knowable 24h after its own decision point, which is still earlier than this
    record's target, so showing it leaks nothing.
    """
    t = record_time(record)
    mid = str(record.get('market_id'))
    own = [r for r in by_market.get(mid, []) if record_time(r) < t]
    siblings = [
        r
        for r in by_event.get(str(record.get('event_id')), [])
        if record_time(r) < t and str(r.get('market_id')) != mid
    ]
    return own, siblings



# ---------------------------------------------------------------------------
# Training. One long prompt per record, plain MSE on the 24h delta.
#
# The shipped recipe emits one prompt per attributed headline and averages the
# per-headline predictions by routing weight. A streaming context already holds
# every headline, so there is nothing left to route over and nothing to weight:
# one sequence, one prediction, one squared error.
# ---------------------------------------------------------------------------


class StreamingRegressionDataset:
    """Tokenised streaming prompts with their delta labels.

    Truncation takes off the FRONT. The prompt ends with the current headlines
    and the `=> 24h change:` anchor that last-token pooling reads, so cutting
    from the right would remove the very thing the regression head pools from.
    Cutting from the left only drops the oldest demonstrations, which is the
    same priority the context builder already packs by.
    """

    def __init__(self, rows: Sequence[dict[str, Any]], tokenizer, max_length: int):
        self.rows = list(rows)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        import torch

        row = self.rows[idx]
        ids = self.tokenizer(row['prompt'], add_special_tokens=False)['input_ids']
        truncated = max(0, len(ids) - self.max_length)
        if truncated:
            ids = ids[-self.max_length :]
        return {
            'input_ids': torch.tensor(ids, dtype=torch.long),
            'label': torch.tensor(float(row['label_delta']), dtype=torch.float),
            'before_price': torch.tensor(float(row['before_price']), dtype=torch.float),
            'market_id': str(row.get('market_id', '')),
            't': int(row.get('t', 0)),
            'n_truncated': truncated,
        }


def collate_streaming(batch: list[dict[str, Any]], pad_token_id: int):
    """LEFT-pad, so every sequence's real last token sits at the same index.

    Right-padding would leave pad tokens after the anchor; last-token pooling
    finds the final non-pad position via the attention mask, so it would still
    work, but left-padding keeps the pooled position identical across the batch
    and matches how these prompts are generated from at inference.
    """
    import torch

    width = max(item['input_ids'].size(0) for item in batch)
    input_ids, attention_mask = [], []
    for item in batch:
        ids = item['input_ids']
        pad = width - ids.size(0)
        input_ids.append(
            torch.cat([torch.full((pad,), pad_token_id, dtype=torch.long), ids])
        )
        attention_mask.append(
            torch.cat([torch.zeros(pad, dtype=torch.long), torch.ones(ids.size(0), dtype=torch.long)])
        )
    return {
        'input_ids': torch.stack(input_ids),
        'attention_mask': torch.stack(attention_mask),
        'labels': torch.stack([item['label'] for item in batch]),
        'before_prices': torch.stack([item['before_price'] for item in batch]),
        'market_ids': [item['market_id'] for item in batch],
        'ts': [item['t'] for item in batch],
    }


def build_streaming_trainer_class():
    """`Trainer` subclass with a plain MSE objective on the delta.

    Built lazily so importing this module does not pull in transformers.
    """
    from transformers import Trainer

    class StreamingTrainer(Trainer):
        def _forward(self, model, inputs):
            import torch

            labels = inputs['labels']
            preds = model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
            ).view(-1)
            loss = torch.nn.functional.mse_loss(preds, labels.to(preds.dtype))
            return loss, preds, labels

        def compute_loss(
            self, model, inputs, return_outputs=False, num_items_in_batch=None
        ):
            loss, preds, _ = self._forward(model, inputs)
            return (loss, preds) if return_outputs else loss

        def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
            import torch

            with torch.no_grad():
                loss, preds, labels = self._forward(model, inputs)
            return loss, preds, labels

    return StreamingTrainer
