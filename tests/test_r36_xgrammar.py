"""Real XGrammar regression tests for EOS boundaries in speculative batches.

Uses a local synthetic vocabulary; no weights, network or GPU are required.
The invalid-token cases ensure terminal guards do not weaken grammar validation.
"""
import pytest
import xgrammar as xgr

from vllm.v1.structured_output.backend_xgrammar import XgrammarGrammar


@pytest.fixture
def grammar():
    info = xgr.TokenizerInfo(
        ['x', '<eos>', 'y'], vocab_type=xgr.VocabType.RAW, stop_token_ids=[1]
    )
    ctx = xgr.GrammarCompiler(info).compile_grammar('root ::= "x"+')
    return XgrammarGrammar(
        vocab_size=3,
        matcher=xgr.GrammarMatcher(ctx),
        ctx=ctx,
    )


def assert_no_terminal_warning(capfd):
    output = capfd.readouterr()
    assert 'trying to accept new token' not in output.out + output.err


@pytest.mark.parametrize('prefix_len', range(1, 7))
def test_validate_eos_anywhere_in_seven_token_window(grammar, capfd, prefix_len):
    prefix = [0] * prefix_len + [1]
    window = prefix + [1] * (7 - len(prefix))
    assert grammar.validate_tokens(window) == prefix
    assert_no_terminal_warning(capfd)
    assert not grammar.matcher.is_terminated()
    assert not grammar.is_terminated()
    assert grammar.num_processed_tokens == 0
    assert grammar.accept_tokens('fresh-after-validation', [0, 1])
    assert grammar.matcher.is_terminated() and grammar.is_terminated()


@pytest.mark.parametrize('prefix_len', range(1, 7))
def test_accept_eos_anywhere_in_seven_token_window(grammar, capfd, prefix_len):
    prefix = [0] * prefix_len + [1]
    window = prefix + [1] * (7 - len(prefix))
    assert grammar.accept_tokens('eos-window', window)
    assert grammar.num_processed_tokens == len(prefix)
    assert grammar.matcher.is_terminated() and grammar.is_terminated()
    assert_no_terminal_warning(capfd)


def test_accept_after_termination_is_idempotent(grammar, capfd):
    assert grammar.accept_tokens('first', [0, 1])
    assert grammar.accept_tokens('already-done', [2, 1, 0])
    assert grammar.validate_tokens([2, 1, 0]) == []
    assert grammar.num_processed_tokens == 2
    assert grammar.matcher.is_terminated() and grammar.is_terminated()
    assert_no_terminal_warning(capfd)


def test_reset_clears_both_termination_states(grammar):
    assert grammar.accept_tokens('first', [0, 1])
    grammar.reset()
    assert not grammar.is_terminated()
    assert not grammar.matcher.is_terminated()
    assert grammar.num_processed_tokens == 0
    assert grammar.accept_tokens('after-reset', [0, 0, 1])
    assert grammar.num_processed_tokens == 3


def test_invalid_token_before_eos_still_rejected(grammar):
    assert not grammar.accept_tokens('invalid-prefix', [2, 0, 1])
    assert grammar.num_processed_tokens == 0
    assert not grammar.is_terminated()
    assert not grammar.matcher.is_terminated()
    assert grammar.accept_tokens('valid-retry', [0, 1])


def test_invalid_token_after_valid_prefix_does_not_consume_suffix(grammar):
    assert not grammar.accept_tokens('invalid-middle', [0, 2, 1])
    assert grammar.num_processed_tokens == 1
    assert not grammar.is_terminated()
    assert not grammar.matcher.is_terminated()
    assert grammar.accept_tokens('valid-continuation', [0, 1])
    assert grammar.num_processed_tokens == 3


def test_draft_validation_rejects_invalid_suffix_and_restores_state(grammar):
    assert grammar.validate_tokens([0, 2, 1]) == [0]
    assert grammar.num_processed_tokens == 0
    assert not grammar.matcher.is_terminated()
    assert grammar.validate_tokens([2, 0, 1]) == []
    assert grammar.accept_tokens('valid-after-bad-draft', [0, 1])


def test_roll_back_eos_then_continue(grammar, capfd):
    assert grammar.accept_tokens('before-rollback', [0, 1])
    grammar.rollback(1)
    assert grammar.num_processed_tokens == 1
    assert not grammar.matcher.is_terminated()
    assert not grammar.is_terminated()
    assert grammar.accept_tokens('after-rollback', [0, 1, 1])
    assert grammar.num_processed_tokens == 3
    assert grammar.matcher.is_terminated() and grammar.is_terminated()
    assert_no_terminal_warning(capfd)
