"""Streaming context construction: no lookahead, and the budget cuts the right end."""

import pytest

from swm.streaming import (
    StreamingContextBuilder,
    anchor_price,
    index_history,
    past_of,
    realised_change,
    record_time,
)

DAY = 86400


def rec(mid, eid, t, hist_last, target_p, news, question='Q?'):
    return {
        'market_id': mid,
        'event_id': eid,
        'question': question,
        'description': '',
        'history': [{'t': t - (16 - i) * DAY, 'p': hist_last} for i in range(16)],
        'target': {'t': t, 'p': target_p},
        'news': [{'title': n, 'description': ''} for n in news],
    }


@pytest.fixture
def stream():
    return [
        rec('m1', 'e1', 100 * DAY, 0.40, 0.45, ['old m1 news']),
        rec('m1', 'e1', 110 * DAY, 0.50, 0.44, ['recent m1 news']),
        rec('m2', 'e1', 105 * DAY, 0.30, 0.38, ['sibling news']),
        rec('m1', 'e1', 120 * DAY, 0.60, 0.70, ['current news']),
        rec('m1', 'e1', 130 * DAY, 0.70, 0.75, ['FUTURE news']),
    ]


# ------------------------------------------------------------------ no lookahead

def test_past_of_excludes_the_present_and_the_future(stream):
    by_market, by_event = index_history(stream)
    current = stream[3]  # m1 @ t=120
    own, siblings = past_of(current, by_market, by_event)
    assert [record_time(r) / DAY for r in own] == [100, 110]
    assert [record_time(r) / DAY for r in siblings] == [105]
    assert all(record_time(r) < record_time(current) for r in own + siblings)


def test_future_news_never_reaches_the_prompt(stream):
    by_market, by_event = index_history(stream)
    current = stream[3]
    own, siblings = past_of(current, by_market, by_event)
    prompt = StreamingContextBuilder(max_tokens=8192).build(current, own, siblings)['prompt']
    assert 'FUTURE news' not in prompt
    assert 'current news' in prompt
    assert 'recent m1 news' in prompt


def test_a_sibling_is_labelled_as_one(stream):
    by_market, by_event = index_history(stream)
    own, siblings = past_of(stream[3], by_market, by_event)
    prompt = StreamingContextBuilder(max_tokens=8192).build(stream[3], own, siblings)['prompt']
    assert 'sibling market' in prompt


def test_siblings_can_be_switched_off(stream):
    by_market, by_event = index_history(stream)
    own, siblings = past_of(stream[3], by_market, by_event)
    built = StreamingContextBuilder(max_tokens=8192, include_siblings=False).build(
        stream[3], own, siblings
    )
    assert built['stats']['sibling_blocks'] == 0
    assert 'sibling news' not in built['prompt']


# ---------------------------------------------------------------------- budget

def test_budget_drops_the_oldest_not_the_current(stream):
    by_market, by_event = index_history(stream)
    own, siblings = past_of(stream[3], by_market, by_event)
    tight = StreamingContextBuilder(max_tokens=120).build(stream[3], own, siblings)
    # The question and the headlines being asked about always survive.
    assert 'current news' in tight['prompt']
    assert tight['stats']['dropped_blocks'] > 0
    assert tight['stats']['own_blocks'] + tight['stats']['sibling_blocks'] < 3


def test_blocks_read_forward_in_time(stream):
    by_market, by_event = index_history(stream)
    own, siblings = past_of(stream[3], by_market, by_event)
    prompt = StreamingContextBuilder(max_tokens=8192).build(stream[3], own, siblings)['prompt']
    assert prompt.index('old m1 news') < prompt.index('recent m1 news')
    assert prompt.index('recent m1 news') < prompt.index('current news')


def test_an_empty_past_still_builds(stream):
    by_market, by_event = index_history(stream)
    first = stream[0]
    own, siblings = past_of(first, by_market, by_event)
    assert own == [] and siblings == []
    built = StreamingContextBuilder(max_tokens=8192).build(first, own, siblings)
    assert 'old m1 news' in built['prompt']
    assert built['stats']['own_blocks'] == 0


# ----------------------------------------------------------------------- label

def test_label_is_the_delta_off_the_last_daily_point(stream):
    current = stream[3]
    assert anchor_price(current) == pytest.approx(0.60)
    assert realised_change(current) == pytest.approx(0.70 - 0.60)


def test_past_blocks_state_their_realised_change(stream):
    by_market, by_event = index_history(stream)
    own, _ = past_of(stream[3], by_market, by_event)
    prompt = StreamingContextBuilder(max_tokens=8192, include_siblings=False).build(
        stream[3], own
    )['prompt']
    assert '-0.060' in prompt  # m1 @ t=110: 0.50 -> 0.44
    assert '+0.050' in prompt  # m1 @ t=100: 0.40 -> 0.45


def test_the_predicted_block_carries_no_answer(stream):
    by_market, by_event = index_history(stream)
    own, siblings = past_of(stream[3], by_market, by_event)
    prompt = StreamingContextBuilder(max_tokens=8192).build(stream[3], own, siblings)['prompt']
    tail = prompt[prompt.index('<-- predict this one'):]
    assert tail.rstrip().endswith('=> 24h change:')
    assert '0.700' not in tail  # the target price is nowhere in the prompt
