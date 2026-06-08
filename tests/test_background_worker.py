from __future__ import annotations

import threading
import unittest

from app.infra.background_worker import submit_background, wait_for_background_tasks


class BackgroundWorkerTests(unittest.TestCase):
    def test_submit_background_runs_task(self) -> None:
        done = threading.Event()

        submit_background(done.set, name="test_set_event")
        wait_for_background_tasks()

        self.assertTrue(done.is_set())

    def test_worker_continues_after_task_failure(self) -> None:
        done = threading.Event()

        def fail() -> None:
            raise RuntimeError("expected failure")

        submit_background(fail, name="test_failure")
        submit_background(done.set, name="test_after_failure")
        wait_for_background_tasks()

        self.assertTrue(done.is_set())


if __name__ == "__main__":
    unittest.main()
