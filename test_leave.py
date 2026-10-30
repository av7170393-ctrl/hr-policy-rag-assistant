import pathlib
import tempfile
import unittest
from datetime import date, timedelta

from app import config, db
from app.agent import tools


def next_monday(weeks_ahead=2):
    d = date.today() + timedelta(weeks=weeks_ahead)
    return d + timedelta(days=(7 - d.weekday()) % 7)


class LeaveTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        config.DB_PATH = pathlib.Path(self.tmp.name) / "test.db"
        db.init_db()
        self.user = db.get_user_by_username("employee")
        self.monday = next_monday()

    def tearDown(self):
        self.tmp.cleanup()

    def apply(self, kind, days_span, start=None):
        s = start or self.monday
        return tools.apply_leave(self.user, kind, s.isoformat(), (s + timedelta(days=days_span)).isoformat())

    def test_valid_casual_leave_deducts_balance(self):
        r = self.apply("casual", 1)
        self.assertTrue(r["ok"])
        self.assertEqual(r["days"], 2)
        self.assertEqual(db.get_balances(self.user["id"])["casual"], 10)

    def test_casual_leave_over_three_days_rejected(self):
        self.assertFalse(self.apply("casual", 3)["ok"])

    def test_not_enough_balance_rejected(self):
        self.assertFalse(self.apply("sick", 14)["ok"])  # 11 working days > 10 sick leaves

    def test_earned_leave_needs_seven_days_notice(self):
        tomorrow = date.today() + timedelta(days=1)
        self.assertFalse(tools.apply_leave(self.user, "earned", tomorrow.isoformat(), tomorrow.isoformat())["ok"])

    def test_sick_leave_over_two_days_adds_certificate_note(self):
        self.assertIn("Medical certificate", self.apply("sick", 2)["note"])

    def test_weekend_only_rejected(self):
        saturday = self.monday - timedelta(days=2)
        self.assertFalse(tools.apply_leave(self.user, "casual", saturday.isoformat(), saturday.isoformat())["ok"])

    def test_reject_refunds_days(self):
        rid = self.apply("casual", 1)["request_id"]
        self.assertTrue(tools.decide_leave(rid, approve=False)["ok"])
        self.assertEqual(db.get_balances(self.user["id"])["casual"], 12)
        self.assertFalse(tools.decide_leave(rid, approve=True)["ok"])  # already decided

    def test_intent_detection(self):
        self.assertTrue(tools.looks_personal("How many casual leaves do I have left?"))
        self.assertTrue(tools.looks_personal("I want to apply for sick leave tomorrow"))
        self.assertFalse(tools.looks_personal("How many casual leaves do I get in a year?"))
        self.assertFalse(tools.looks_personal("What is the notice period?"))

    def test_auth_and_analytics(self):
        self.assertIsNotNone(db.authenticate("hr", "hr123"))
        self.assertIsNone(db.authenticate("hr", "bad"))
        db.add_message(self.user["id"], "q?", "a", [], "escalated", 100, None)
        self.assertEqual(db.analytics()["total_questions"], 1)


if __name__ == "__main__":
    unittest.main()
