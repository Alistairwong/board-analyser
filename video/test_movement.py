"""Unit tests for movement.py, using synthetic landmark tracks (no video needed).

Run with:
    cd video && python -m unittest test_movement.py
"""
import unittest

import movement as mv

# A tiny synthetic climb: 4 holds in a vertical line, one inch apart in x so
# nearest_hold has no ambiguity, HOLD_TOLERANCE (4in) apart in y.
HOLDS = [
    {"hole_id": 1, "x": 0.0, "y": 0.0, "role_name": "start"},
    {"hole_id": 2, "x": 1.0, "y": 10.0, "role_name": "middle"},
    {"hole_id": 3, "x": 0.0, "y": 20.0, "role_name": "middle"},
    {"hole_id": 4, "x": 1.0, "y": 30.0, "role_name": "finish"},
]
HOLDS_BY_ID = {h["hole_id"]: h for h in HOLDS}


def frame(t, **points):
    return {"t": t, "points": points}


class NearestHoldTests(unittest.TestCase):
    def test_within_tolerance_matches(self):
        self.assertEqual(mv.nearest_hold((0.1, 0.1), HOLDS), 1)

    def test_beyond_tolerance_is_none(self):
        self.assertIsNone(mv.nearest_hold((0.0, 5.0), HOLDS))

    def test_picks_the_closer_of_two(self):
        # (0, 10) is 1.0 from hold 2 (1, 10) and 10.0 from hold 1 (0, 0).
        self.assertEqual(mv.nearest_hold((0.0, 10.0), HOLDS), 2)


class AssignHoldsTests(unittest.TestCase):
    def test_assigns_each_limb_independently(self):
        track = [frame(0.0, left_wrist=(0.0, 0.0), right_wrist=(1.0, 10.0))]
        assignments = mv.assign_holds(track, HOLDS)
        self.assertEqual(assignments[0]["assign"]["left_wrist"], 1)
        self.assertEqual(assignments[0]["assign"]["right_wrist"], 2)
        self.assertIsNone(assignments[0]["assign"]["left_foot"])

    def test_missing_landmark_is_none(self):
        track = [frame(0.0)]   # no points at all this frame (tracking lost)
        assignments = mv.assign_holds(track, HOLDS)
        self.assertTrue(all(v is None for v in assignments[0]["assign"].values()))


class DetectMovesTests(unittest.TestCase):
    def _assignments(self, sequence, limb="left_wrist"):
        """Build assignments for one limb from a list of (t, hole_id)."""
        return [{"t": t, "assign": {l: (hole_id if l == limb else None) for l in mv.LIMBS}}
                for t, hole_id in sequence]

    def test_stable_change_is_a_move(self):
        seq = [(0.0, 1), (0.1, 1), (0.2, 1), (0.3, 2), (0.4, 2), (0.5, 2)]
        moves = mv.detect_moves(self._assignments(seq), min_run=3)
        self.assertEqual(len(moves), 1)
        self.assertEqual(moves[0], {"limb": "left_wrist", "from_hold": 1, "to_hold": 2,
                                     "t": 0.3, "duration_since_prev": None})

    def test_short_blip_is_not_a_move(self):
        # Hand passes near hold 2 for one frame on the way to gripping hold 1 properly.
        seq = [(0.0, 1), (0.1, 2), (0.2, 1), (0.3, 1), (0.4, 1)]
        moves = mv.detect_moves(self._assignments(seq), min_run=3)
        self.assertEqual(moves, [])

    def test_no_move_when_assignment_never_changes(self):
        seq = [(0.0, 1), (0.1, 1), (0.2, 1)]
        moves = mv.detect_moves(self._assignments(seq), min_run=3)
        self.assertEqual(moves, [])

    def test_multiple_moves_get_durations_and_stay_sorted(self):
        seq = [(0.0, 1), (0.1, 1), (0.2, 1),
               (1.0, 2), (1.1, 2), (1.2, 2),
               (3.0, 3), (3.1, 3), (3.2, 3)]
        moves = mv.detect_moves(self._assignments(seq), min_run=3)
        self.assertEqual([m["to_hold"] for m in moves], [2, 3])
        self.assertIsNone(moves[0]["duration_since_prev"])
        self.assertAlmostEqual(moves[1]["duration_since_prev"], 2.0)


class ComputeMetricsTests(unittest.TestCase):
    def test_empty_track(self):
        self.assertEqual(mv.compute_metrics([], []), {
            "duration_s": 0.0, "move_count": 0, "moves_per_min": 0.0,
            "rest_time_s": 0.0, "longest_pause_s": 0.0, "longest_pause_hold": None,
        })

    def test_typical_attempt(self):
        track = [frame(0.0), frame(12.0)]
        moves = [
            {"limb": "left_wrist", "from_hold": 1, "to_hold": 2, "t": 2.0, "duration_since_prev": None},
            {"limb": "right_wrist", "from_hold": 2, "to_hold": 3, "t": 4.0, "duration_since_prev": 2.0},
            {"limb": "left_wrist", "from_hold": 2, "to_hold": 4, "t": 12.0, "duration_since_prev": 8.0},
        ]
        metrics = mv.compute_metrics(moves, track)
        self.assertEqual(metrics["duration_s"], 12.0)
        self.assertEqual(metrics["move_count"], 3)
        self.assertEqual(metrics["longest_pause_s"], 8.0)
        self.assertEqual(metrics["longest_pause_hold"], 2)   # from_hold of the longest-gap move
        self.assertGreater(metrics["rest_time_s"], 0.0)


class ClassifyStylesTests(unittest.TestCase):
    def test_still_hips_is_static(self):
        track = [frame(t, left_hip=(0.0, 50.0), right_hip=(2.0, 50.0)) for t in (0.7, 0.9, 1.0, 1.1, 1.3)]
        moves = [{"limb": "left_wrist", "from_hold": 1, "to_hold": 2, "t": 1.0, "duration_since_prev": None}]
        mv.classify_styles(moves, track)
        self.assertEqual(moves[0]["style"], "static")

    def test_launching_hips_is_dynamic(self):
        # Hips rise 15in in 0.2s around the move = 75in/s, well above the threshold.
        track = [
            frame(0.8, left_hip=(0.0, 50.0), right_hip=(2.0, 50.0)),
            frame(0.9, left_hip=(0.0, 35.0), right_hip=(2.0, 35.0)),
            frame(1.0, left_hip=(0.0, 20.0), right_hip=(2.0, 20.0)),
        ]
        moves = [{"limb": "right_wrist", "from_hold": 1, "to_hold": 2, "t": 0.9, "duration_since_prev": None}]
        mv.classify_styles(moves, track)
        self.assertEqual(moves[0]["style"], "dynamic")

    def test_foot_moves_are_not_classified(self):
        moves = [{"limb": "left_foot", "from_hold": 1, "to_hold": 2, "t": 1.0, "duration_since_prev": None}]
        mv.classify_styles(moves, [])
        self.assertIsNone(moves[0]["style"])


class FlagHesitationsTests(unittest.TestCase):
    def test_long_pause_is_flagged(self):
        moves = [
            {"t": 0.0, "duration_since_prev": None},
            {"t": 1.0, "duration_since_prev": 1.0},
            {"t": 2.0, "duration_since_prev": 1.0},
            {"t": 8.0, "duration_since_prev": 6.0},
        ]
        mv.flag_hesitations(moves)
        self.assertEqual([m["hesitated"] for m in moves], [False, False, False, True])

    def test_uniform_pace_has_no_hesitation(self):
        moves = [{"t": t, "duration_since_prev": (None if t == 0 else 1.0)} for t in (0.0, 1.0, 2.0, 3.0)]
        mv.flag_hesitations(moves)
        self.assertTrue(all(not m["hesitated"] for m in moves))


class FlagHipsTests(unittest.TestCase):
    def test_offset_hip_is_flagged_out(self):
        assignments = [{"t": 1.0, "assign": {"left_wrist": 4, "right_wrist": 1,
                                              "left_ankle": None, "right_ankle": None}}]
        track = [frame(1.0, left_hip=(20.0, 15.0), right_hip=(20.0, 15.0))]
        moves = [{"limb": "left_wrist", "from_hold": 2, "to_hold": 4, "t": 1.0, "duration_since_prev": None}]
        mv.flag_hips(moves, assignments, track, HOLDS_BY_ID)
        self.assertTrue(moves[0]["hips_out"])
        self.assertAlmostEqual(moves[0]["hips_offset"], 20.0 - HOLDS_BY_ID[1]["x"])

    def test_centred_hip_is_not_flagged(self):
        assignments = [{"t": 1.0, "assign": {"left_wrist": 4, "right_wrist": 1,
                                              "left_ankle": None, "right_ankle": None}}]
        track = [frame(1.0, left_hip=(0.5, 15.0), right_hip=(0.5, 15.0))]
        moves = [{"limb": "left_wrist", "from_hold": 2, "to_hold": 4, "t": 1.0, "duration_since_prev": None}]
        mv.flag_hips(moves, assignments, track, HOLDS_BY_ID)
        self.assertFalse(moves[0]["hips_out"])


class AttemptOutcomeTests(unittest.TestCase):
    def test_hand_on_finish_hold_is_topped(self):
        moves = [{"limb": "left_wrist", "from_hold": 2, "to_hold": 4, "t": 5.0}]
        self.assertEqual(mv.attempt_outcome(moves, HOLDS), "topped")

    def test_no_finish_hold_is_unknown(self):
        moves = [{"limb": "left_wrist", "from_hold": 1, "to_hold": 2, "t": 5.0}]
        self.assertEqual(mv.attempt_outcome(moves, HOLDS), "unknown")

    def test_foot_on_finish_hold_does_not_count(self):
        moves = [{"limb": "left_foot", "from_hold": 2, "to_hold": 4, "t": 5.0}]
        self.assertEqual(mv.attempt_outcome(moves, HOLDS), "unknown")


class AnalyseEndToEndTests(unittest.TestCase):
    def test_full_climb_is_recognised_as_topped(self):
        """Both hands walk up holds 1->2->3->4 (finish); feet stay put."""
        track = []
        for t in (0.0, 0.1, 0.2):
            track.append(frame(t, left_wrist=(0.0, 0.0), right_wrist=(0.0, 0.0),
                               left_hip=(0.5, -10.0), right_hip=(0.5, -10.0)))
        for t in (1.0, 1.1, 1.2):
            track.append(frame(t, left_wrist=(1.0, 10.0), right_wrist=(0.0, 0.0),
                               left_hip=(0.5, -5.0), right_hip=(0.5, -5.0)))
        for t in (2.0, 2.1, 2.2):
            track.append(frame(t, left_wrist=(1.0, 10.0), right_wrist=(0.0, 20.0),
                               left_hip=(0.5, 0.0), right_hip=(0.5, 0.0)))
        for t in (3.0, 3.1, 3.2):
            track.append(frame(t, left_wrist=(1.0, 30.0), right_wrist=(0.0, 20.0),
                               left_hip=(0.5, 5.0), right_hip=(0.5, 5.0)))

        result = mv.analyse(track, HOLDS)
        self.assertEqual(result["attempt"]["outcome"], "topped")
        self.assertEqual(result["metrics"]["move_count"], len(result["moves"]))
        self.assertGreaterEqual(result["metrics"]["move_count"], 3)
        self.assertEqual(result["attempt"]["start_s"], 0.0)
        self.assertEqual(result["attempt"]["end_s"], 3.2)


if __name__ == "__main__":
    unittest.main()
