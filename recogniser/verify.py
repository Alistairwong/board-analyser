"""Independent cross-checks that the recognised climb really is the right one.

Each check fails for different reasons (camera colours, the pose tracker), so when several
agree the answer is much more trustworthy than any one alone. Pure functions over plain data,
so they're tested with synthetic tracks (see test_recognise.py).

A check is {"name", "status", "detail", "source"}: status is one of AGREE / DISAGREE / INCONCLUSIVE / NA, and
source is LEDS or BODY. Checks from the same source share a failure mode, so only agreement across
different sources counts as confirmation.
"""
from movement import FOOT_LIMBS, HAND_LIMBS, HOLD_TOLERANCE, _stable_runs, assign_holds, nearest_hold

AGREE, DISAGREE, INCONCLUSIVE, NA = "agrees", "disagrees", "inconclusive", "not available"
LEDS, BODY = "lit holds", "body pose"
SPLIT_NAME = "Same answer from each half of the video"
EMPTY_NAME = "Empty-board image and the whole video agree"
BODY_NAME = "Climber's hands and feet stay on the climb"
START_NAME = "Starts on the start holds, finishes on the finish"
MIN_TRACKED = 15          # limb-frames needed before the body checks say anything
ON_ROUTE = 0.5            # share of limb-frames on the climb's holds that counts as "on the route"
CLEAR_WIN = 0.1           # another climb must beat the recognised one by this much to disagree


SOURCES = {EMPTY_NAME: LEDS, SPLIT_NAME: LEDS, BODY_NAME: BODY, START_NAME: BODY}   # so a "not available" check still says where it comes from


def check(name, status, detail, images=None):
    """images: [{"url", "caption"}] pictures this check looked at, added by the server."""
    return {"name": name, "status": status, "detail": detail, "source": SOURCES.get(name, LEDS), "images": images or []}


def led_check(best, verdict, ties):
    """The main recognition: how well the lit holds match, and how far ahead of the runner-up."""
    if not best:
        return check("Lit holds match the climb", DISAGREE, "No climb matched the lit holds.")
    text = f"{best['jaccard']:.0%} of the holds overlap"
    if verdict in ("confident", "likely"):
        return check("Lit holds match the climb", AGREE, f"{text} ({verdict}).")
    if verdict == "tied":
        return check("Lit holds match the climb", INCONCLUSIVE, f"{text}, but {ties} other climb(s) match equally well.")
    return check("Lit holds match the climb", INCONCLUSIVE, f"{text}, which isn't enough to be sure.")


def split_half(halves, best):
    """halves: [(uuid, mirrored, jaccard, name) or None] for the first and second half of the video."""
    name = SPLIT_NAME
    if any(h is None for h in halves):
        return check(name, NA, "One half of the video had no lit holds to compare.")
    same = [h[0] == best["uuid"] and h[1] == best["mirrored"] for h in halves]
    detail = "; ".join(f"{label} half: {h[3]} ({h[2]:.0%})" for label, h in zip(("first", "second"), halves))
    if all(same):
        return check(name, AGREE, detail)
    if any(same):
        return check(name, INCONCLUSIVE, detail + ". A climber blocking the LEDs for a while can do this.")
    return check(name, DISAGREE, detail)


def empty_vs_whole(whole, best):
    """whole: (uuid, mirrored, jaccard, name) from the whole-video detection, or None. The empty-board image
    is the better reading, so a difference here is only "inconclusive": the climber probably hid some LEDs."""
    if whole is None:
        return check(EMPTY_NAME, NA, "Reading the whole video found no lit holds to compare.")
    detail = f"median picture of the whole video: {whole[3]} ({whole[2]:.0%})"
    if whole[0] == best["uuid"] and whole[1] == best["mirrored"]:
        return check(EMPTY_NAME, AGREE, detail)
    return check(EMPTY_NAME, INCONCLUSIVE, detail + ". A climber blocking LEDs during the video can do this.")


def route_score(track, holds):
    """Share of tracked limb-frames that sit on the climb's holds: hands on hand holds, feet on any hold."""
    hand = [h for h in holds if h.get("role_name") != "foot"]
    near = total = 0
    for f in track:
        for limb, pt in f["points"].items():
            if limb in HAND_LIMBS:
                pool = hand
            elif limb in FOOT_LIMBS:
                pool = holds
            else:
                continue
            total += 1
            near += nearest_hold(pt, pool, HOLD_TOLERANCE) is not None
    return (near / total if total else 0.0), total


def body_on_route(track, candidates, best_uuid):
    """candidates: [{"uuid", "name", "holds"}]. Does the climber's body sit on the recognised climb's holds
    at least as well as on any runner-up's? ponytail: coverage-based (a climb with many holds scores a
    little higher by chance); a per-hold likelihood would fix that if it ever misleads."""
    name = BODY_NAME
    scores = {c["uuid"]: route_score(track, c["holds"]) for c in candidates}
    ours, total = scores[best_uuid]
    if total < MIN_TRACKED:
        return check(name, NA, "The pose tracker didn't follow the climber for long enough.")
    rival = max((c for c in candidates if c["uuid"] != best_uuid), key=lambda c: scores[c["uuid"]][0], default=None)
    detail = f"{ours:.0%} of hand/foot positions were on this climb's holds"
    if rival and scores[rival["uuid"]][0] > ours + CLEAR_WIN:
        return check(name, DISAGREE, f"{detail}, but {scores[rival['uuid']][0]:.0%} were on '{rival['name']}'.")
    if ours >= ON_ROUTE and (not rival or ours >= scores[rival["uuid"]][0] - 0.02):
        return check(name, AGREE, detail + (f" (next best: '{rival['name']}' {scores[rival['uuid']][0]:.0%})." if rival else "."))
    return check(name, INCONCLUSIVE, detail + ".")


def start_and_finish(track, holds, topped):
    """Hands should begin on a start hold and (if the climb was completed) end on a finish hold."""
    name = START_NAME
    role = {h["hole_id"]: h.get("role_name") for h in holds}
    assignments = assign_holds(track, holds)
    # each hand's first stable hold (runs on no hold at all, e.g. tracking lost, don't count)
    first = [h for limb in HAND_LIMBS
             if (h := next((hold for _, hold in _stable_runs(assignments, limb) if hold is not None), None)) is not None]
    if not first:
        return check(name, NA, "Couldn't see the climber's hands on any hold.")
    started = any(role.get(h) == "start" for h in first)
    if started and topped:
        return check(name, AGREE, "Hands began on a start hold and ended on a finish hold.")
    if started:
        return check(name, AGREE, "Hands began on a start hold. The attempt didn't reach the finish.")
    if topped:
        return check(name, AGREE, "Hands ended on a finish hold; the start wasn't clearly seen.")
    return check(name, DISAGREE, "Hands didn't start on a start hold or reach a finish hold of this climb.")


def best_route_frame(track, holds):
    """(frame index, limbs on the climb, limbs tracked) for the frame where most tracked limbs sit on the climb's holds."""
    hand = [h for h in holds if h.get("role_name") != "foot"]
    best = None
    for i, f in enumerate(track):
        on = total = 0
        for limb, pt in f["points"].items():
            if limb in HAND_LIMBS:
                pool = hand
            elif limb in FOOT_LIMBS:
                pool = holds
            else:
                continue
            total += 1
            on += nearest_hold(pt, pool, HOLD_TOLERANCE) is not None
        if total and (best is None or (on, total) > (best[1], best[2])):
            best = (i, on, total)
    return best


def start_finish_frames(track, holds):
    """(first frame with a hand on a start hold, last frame with a hand on a finish hold); None where never."""
    role = {h["hole_id"]: h.get("role_name") for h in holds}
    assignments = assign_holds(track, holds)
    on = lambda a, r: any(role.get(a["assign"][limb]) == r for limb in HAND_LIMBS)
    start = next((i for i, a in enumerate(assignments) if on(a, "start")), None)
    finish = next((i for i in range(len(assignments) - 1, -1, -1) if on(assignments[i], "finish")), None)
    return start, finish


def summary(checks):
    """One line for the top of the results. "confirmed" needs agreement from both the lit holds and the body,
    with nothing disagreeing; two LED-based checks agreeing only rules out a fluke, not a colour problem."""
    live = [c for c in checks if c["status"] != NA]
    agree = [c for c in live if c["status"] == AGREE]
    disagree = [c for c in live if c["status"] == DISAGREE]
    sources = {c["source"] for c in agree}
    count = f"{len(agree)} of {len(live)} checks agree"
    if disagree:
        level, text = "double-check", f"{count}, {len(disagree)} disagree. Worth a second look."
    elif len(sources) >= 2:
        level, text = "confirmed", f"{count}, from both the lit holds and the climber's body."
    else:
        level, text = "unconfirmed", f"{count}, but only from the {' and '.join(sorted(sources)) or 'lit holds'}, so nothing independent yet."
    return {"level": level, "text": text, "checks": checks}
