"""The sessions tests (test_sessions.py) on Oracle 23ai: the same scenarios, the other SQL dialect.

Marked `oracle`, so they run only with ORACLE_TEST_DSN set (see conftest's oracle_memory).
"""

import pytest

from tests import test_sessions as pg

pytestmark = pytest.mark.oracle


@pytest.fixture
def memory(oracle_memory):
    return oracle_memory


test_append_is_ordered_and_a_retry_stores_nothing_twice = pg.test_append_is_ordered_and_a_retry_stores_nothing_twice
test_a_note_is_logged_delivered_until_acked_and_never_repeated = (
    pg.test_a_note_is_logged_delivered_until_acked_and_never_repeated
)
test_a_turn_end_saves_the_turn_into_the_document_once = pg.test_a_turn_end_saves_the_turn_into_the_document_once
test_a_save_that_died_after_submitting_resubmits_the_same_operation = (
    pg.test_a_save_that_died_after_submitting_resubmits_the_same_operation
)
test_an_idle_session_saves_what_is_unsaved_and_retain_off_never_does = (
    pg.test_an_idle_session_saves_what_is_unsaved_and_retain_off_never_does
)
test_sessions_are_bank_scoped_and_listed_newest_first = pg.test_sessions_are_bank_scoped_and_listed_newest_first
test_a_check_waits_for_the_session_to_settle_on_a_queue = pg.test_a_check_waits_for_the_session_to_settle_on_a_queue
test_http_routes = pg.test_http_routes
