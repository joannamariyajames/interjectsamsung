"""Offline India gazetteer, routing and understanding of disfluent driving speech."""

from __future__ import annotations

import pytest
from app.drive.geo import describe_duration, gazetteer, plan_route
from app.drive.understanding import understand


def _dest(text: str) -> str | None:
    u = understand(text)
    if u.destination is None:
        return None
    return u.destination.place.name if u.destination.place else f"saved:{u.destination.saved_label}"


# -- gazetteer ---------------------------------------------------------------

@pytest.mark.parametrize(
    ("spoken", "expected"),
    [
        ("Jaipur", "Jaipur"), ("bangalore", "Bengaluru"), ("Bombay", "Mumbai"), ("poona", "Pune"),
        ("mysore", "Mysuru"), ("gurgaon", "Gurugram"), ("new delhi", "Delhi"), ("cochin", "Kochi"),
        ("coimbatur", "Coimbatore"), ("jaipor", "Jaipur"),  # near-miss spellings from speech recognition
    ],
)
def test_lookup_by_name_alternate_and_near_miss(spoken: str, expected: str) -> None:
    place = gazetteer().lookup(spoken)
    assert place is not None and place.name == expected


def test_gazetteer_covers_india_not_one_city() -> None:
    gz = gazetteer()
    assert len(gz.places) > 5000
    assert len({p.state for p in gz.places}) > 25


def test_ambiguous_names_prefer_the_larger_place_unless_a_state_is_given() -> None:
    gz = gazetteer()
    assert gz.lookup("Aurangabad").state == "Maharashtra"
    assert gz.lookup("Aurangabad", state="Bihar").state == "Bihar"


def test_state_names_route_to_the_state() -> None:
    goa = gazetteer().lookup("Goa")
    assert goa is not None and goa.state == "Goa" and goa.label.startswith("Goa")


def test_unknown_place_is_not_invented() -> None:
    assert gazetteer().lookup("Zzyzx") is None


# -- routing -------------------------------------------------------------------

def test_intercity_route_is_realistic() -> None:
    gz = gazetteer()
    route = plan_route(gz.lookup("Delhi"), gz.lookup("Jaipur"))
    assert 250 < route.distance_km < 330  # the real road distance is about 280 km
    assert 4 * 60 <= route.eta_min <= 6 * 60
    assert route.via  # a town in the corridor


def test_stops_add_time_and_sit_near_a_real_town() -> None:
    gz = gazetteer()
    plain = plan_route(gz.lookup("Mumbai"), gz.lookup("Pune"))
    with_stops = plan_route(gz.lookup("Mumbai"), gz.lookup("Pune"), ("fuel", "food"))
    assert with_stops.eta_min > plain.eta_min
    assert [s["kind"] for s in with_stops.stops] == ["fuel", "food"]
    assert all(gz.lookup(s["near"]) is not None for s in with_stops.stops)


def test_avoiding_highways_is_slower_and_routes_are_deterministic() -> None:
    gz = gazetteer()
    a, b = gz.lookup("Chennai"), gz.lookup("Bengaluru")
    assert plan_route(a, b, (), True).eta_min > plan_route(a, b).eta_min
    assert plan_route(a, b).to_dict() == plan_route(a, b).to_dict()


def test_describe_duration() -> None:
    assert describe_duration(45) == "45 minutes"
    assert describe_duration(60) == "1 hour"
    assert describe_duration(152) == "2 hours 32 minutes"


# -- understanding -------------------------------------------------------------

@pytest.mark.parametrize(
    ("spoken", "destination"),
    [
        ("Take me to Delhi", "Delhi"),
        ("take me to, um, Delhi - actually no, Jaipur", "Jaipur"),
        ("drive to Bombay, um, no wait, Poona", "Pune"),
        ("Jaipur... wait, make it Udaipur instead", "Udaipur"),
        ("not Pune, Nashik", "Nashik"),
        ("go to Chennai instead of Bangalore", "Chennai"),
        ("head to Navi Mumbai please", "Navi Mumbai"),
        ("i want to reach Mysore by evening", "Mysuru"),
        ("okay so, uh, Jaipur", "Jaipur"),
        ("take me to jaipor", "Jaipur"),
        ("go to Than", "Than"),  # a real town that is also a word: fine after "to"
    ],
)
def test_final_destination_after_disfluency_and_self_correction(spoken: str, destination: str) -> None:
    assert _dest(spoken) == destination


@pytest.mark.parametrize("spoken", ["I got to go to Jaipur", "may I go to Jaipur"])
def test_town_names_that_are_ordinary_words_are_not_destinations(spoken: str) -> None:
    u = understand(spoken)
    assert u.destination.place.name == "Jaipur"
    assert u.abandoned == []  # never "forget Got" / "forget May"


def test_abandoned_destination_is_reported_not_acted_on() -> None:
    u = understand("take me to Agra, actually no, Jaipur")
    assert u.destination.place.name == "Jaipur"
    assert [m.place.name for m in u.abandoned] == ["Agra"]
    assert u.corrected


def test_origin_and_destination() -> None:
    u = understand("from Mumbai to Goa and uh stop for fuel on the way")
    assert u.origin.place.name == "Mumbai"
    assert u.destination.place.state == "Goa"
    assert u.add_stops == ["fuel"]
    assert understand("I'm in Pune, take me to Mumbai").origin.place.name == "Pune"


@pytest.mark.parametrize(
    ("spoken", "added", "removed", "abandoned"),
    [
        ("I need to get fuel", ["fuel"], [], []),
        ("find a charger and somewhere to eat", ["charging", "food"], [], []),
        ("stop for coffee, actually no, fuel", ["fuel"], [], ["coffee"]),
        ("skip the coffee stop", [], ["coffee"], []),
        ("no coffee", [], ["coffee"], []),
    ],
)
def test_stops_added_dropped_and_corrected(spoken: str, added: list[str], removed: list[str], abandoned: list[str]) -> None:
    u = understand(spoken)
    assert (u.add_stops, u.remove_stops, u.abandoned_stops) == (added, removed, abandoned)


def test_saved_places() -> None:
    u = understand("my office is in Noida")
    assert (u.save_label, u.save_place.place.name, u.destination) == ("office", "Noida", None)
    assert understand("set home to Pune").save_label == "home"
    u = understand("save this as gym")
    assert (u.save_label, u.save_place, u.destination) == ("gym", None, None)
    assert _dest("take me to the office") == "saved:office"
    assert _dest("take me home") == "saved:home"


def test_unknown_destination_is_flagged() -> None:
    u = understand("take me to Zzyzx")
    assert u.destination is None and u.unresolved == "zzyzx"


@pytest.mark.parametrize(
    ("spoken", "flag"),
    [
        ("how long will it take", "eta"), ("when will we reach", "eta"), ("how far is it", "distance"),
        ("where are we going", "status"), ("cancel the navigation", "cancel"), ("go on", "resume"),
        ("say that again", "repeat"), ("hello, what can you do", "greeting"),
    ],
)
def test_questions_and_commands(spoken: str, flag: str) -> None:
    assert getattr(understand(spoken), flag) is True


def test_route_preferences() -> None:
    assert understand("avoid highways").avoid_highways is True
    assert understand("no tolls please").avoid_highways is True
    assert understand("highways are fine").avoid_highways is False
    assert understand("take me to Delhi").avoid_highways is None
