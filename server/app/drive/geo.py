"""Offline geography for the in-car assistant: every Indian city and town with
1,000+ people (GeoNames, CC BY 4.0 - see data/build_places.py), looked up by
name, alternate name or a near-miss spelling, and routed between.

Routes are estimates from coordinates, not turn-by-turn directions: road
distance = great-circle distance x a road factor, time from city/highway
speeds, the "via" town is the largest place in the corridor near the middle,
and stops are placed near a real town at a sensible point along the way. That
is honest for city-to-city driving without a map service, and it is fully
deterministic - the same request always gives the same route.
"""

from __future__ import annotations

import difflib
import math
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

DATA = Path(__file__).with_name("data") / "india_places.tsv"


@dataclass(frozen=True)
class Place:
    place_id: str
    name: str
    state: str
    lat: float
    lon: float
    population: int

    @property
    def label(self) -> str:
        if self.place_id.startswith("state:") or not self.state or self.state == self.name:
            return self.name
        return f"{self.name}, {self.state}"

    def to_dict(self) -> dict[str, object]:
        return {"id": self.place_id, "name": self.name, "state": self.state, "label": self.label,
                "lat": self.lat, "lon": self.lon}


def normalise(text: str) -> str:
    text = text.lower().replace("&", " and ")
    text = re.sub(r"[^a-z0-9' ]+", " ", text)
    text = re.sub(r"\b(the|city of|town of)\b", " ", text)
    return re.sub(r"\s+", " ", text).strip()


class Gazetteer:
    """Name -> places index over the bundled dataset."""

    def __init__(self, path: Path = DATA) -> None:
        self.places: list[Place] = []
        self.by_name: dict[str, list[Place]] = {}
        self.states: set[str] = set()
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("#") or line.startswith("geonameid\t"):
                continue
            gid, name, state, lat, lon, pop, alternates = (line.split("\t") + [""] * 7)[:7]
            place = Place(gid, name, state, float(lat), float(lon), int(pop or 0))
            self.places.append(place)
            if state:
                self.states.add(normalise(state))
            for key in {normalise(name), *(normalise(a) for a in alternates.split("|") if a)}:
                if key:
                    self.by_name.setdefault(key, []).append(place)
        for key in self.by_name:  # most populous first: "Aurangabad" means the big one
            self.by_name[key].sort(key=lambda p: -p.population)
        # A state name ("Goa", "Kerala") routes to that state's most populous place.
        biggest: dict[str, Place] = {}
        for place in self.places:  # the data file is sorted by population
            if place.state and place.state not in biggest:
                biggest[place.state] = place
        for state, main in biggest.items():
            key = normalise(state)
            if key and key not in self.by_name:
                self.by_name[key] = [Place(f"state:{key}", f"{state} ({main.name})", state, main.lat, main.lon, main.population)]
        self.max_words = max(len(k.split()) for k in self.by_name)

    def exact(self, phrase: str, state: str | None = None) -> Place | None:
        candidates = self.by_name.get(normalise(phrase), [])
        if state:
            in_state = [p for p in candidates if normalise(p.state) == normalise(state)]
            candidates = in_state or candidates
        return candidates[0] if candidates else None

    def fuzzy(self, phrase: str) -> Place | None:
        """A near-miss spelling ("jaipor", "coimbatur") - speech recognisers mangle names."""
        key = normalise(phrase)
        if len(key) < 5:
            return None
        match = difflib.get_close_matches(key, self.by_name.keys(), n=1, cutoff=0.8)
        return self.by_name[match[0]][0] if match else None

    def lookup(self, phrase: str, state: str | None = None) -> Place | None:
        return self.exact(phrase, state) or self.fuzzy(phrase)


@lru_cache(maxsize=1)
def gazetteer() -> Gazetteer:
    return Gazetteer()


# -- routing ----------------------------------------------------------------

def km_between(a: Place, b: Place) -> float:
    return _haversine(a.lat, a.lon, b.lat, b.lon)


def _haversine(a_lat: float, a_lon: float, b_lat: float, b_lon: float) -> float:
    p1, p2 = math.radians(a_lat), math.radians(b_lat)
    dp, dl = p2 - p1, math.radians(b_lon - a_lon)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * 6371.0 * math.asin(math.sqrt(h))


STOP_KINDS = ("fuel", "charging", "food", "coffee", "rest")
STOP_LABEL = {"fuel": "fuel", "charging": "EV charging", "food": "a meal", "coffee": "coffee", "rest": "a rest break"}
STOP_MINUTES = {"fuel": 8, "charging": 30, "food": 30, "coffee": 12, "rest": 10}
STOP_AT_FRACTION = {"fuel": 0.5, "charging": 0.55, "food": 0.45, "coffee": 0.3, "rest": 0.6}


@dataclass
class Route:
    origin: Place
    destination: Place
    distance_km: float
    eta_min: int
    via: str
    avoid_highways: bool
    stops: list[dict[str, object]] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "origin": self.origin.to_dict(),
            "destination": self.destination.to_dict(),
            "distance_km": self.distance_km,
            "eta_min": self.eta_min,
            "via": self.via,
            "avoid_highways": self.avoid_highways,
            "stops": list(self.stops),
        }


def _point_along(a: Place, b: Place, fraction: float) -> tuple[float, float]:
    return a.lat + (b.lat - a.lat) * fraction, a.lon + (b.lon - a.lon) * fraction


def _town_near(lat: float, lon: float, radius_km: float, exclude: set[str]) -> Place | None:
    """The most populous town within radius_km of a point (the likeliest place to stop)."""
    best: Place | None = None
    for p in gazetteer().places:  # sorted by population, so the first hit is the biggest
        if p.place_id in exclude:
            continue
        if abs(p.lat - lat) > radius_km / 100 or abs(p.lon - lon) > radius_km / 100:
            continue
        if _haversine(lat, lon, p.lat, p.lon) <= radius_km:
            best = p
            break
    return best


def plan_route(origin: Place, destination: Place, stop_kinds: tuple[str, ...] = (), avoid_highways: bool = False) -> Route:
    straight = km_between(origin, destination)
    road_factor = 1.35 if straight < 40 else 1.22
    distance = straight * road_factor
    endpoints = {origin.place_id, destination.place_id}

    if distance < 35:
        speed, via = 24.0, ""
    else:
        speed = 42.0 if avoid_highways else 58.0
        mid = _town_near(*_point_along(origin, destination, 0.5), max(20.0, straight * 0.15), endpoints)
        via = mid.name if mid else ""

    stops: list[dict[str, object]] = []
    extra_min = 0
    for kind in stop_kinds:
        fraction = STOP_AT_FRACTION[kind] if distance >= 35 else 0.5
        near = _town_near(*_point_along(origin, destination, fraction), max(15.0, straight * 0.12), endpoints)
        near = near or (origin if fraction < 0.5 else destination)
        stops.append({"kind": kind, "label": STOP_LABEL[kind], "near": near.name,
                      "km_from_start": round(distance * fraction, 1)})
        extra_min += STOP_MINUTES[kind]

    minutes = distance / speed * 60 + extra_min
    return Route(origin, destination, round(distance, 1), max(1, round(minutes)), via, avoid_highways, stops)


def describe_duration(minutes: int) -> str:
    if minutes < 60:
        return f"{minutes} minutes"
    hours, mins = divmod(minutes, 60)
    hours_text = f"{hours} hour" + ("s" if hours != 1 else "")
    return hours_text if mins < 5 else f"{hours_text} {mins} minutes"
