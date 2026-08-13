"""
deterministic_scenario.py
─────────────────────────
Network definition for the deterministic velocity-profile evaluation scenario.

Scenario design:
  - Ego vehicle spawns on the WEST lane (E#L-X) at t = 20 s.
  - Ego can go either NORTH (E#X-T) or EAST (E#X-R) — route is fixed
    per episode (alternating or explicitly set by the environment).
  - Background vehicles spawn on the SOUTH lane (E#D-X) at a fixed
    period, all going NORTH (E#X-T).
  - No other inflows, no probabilistic route assignment.

Edge ID convention (matches existing 100m networks):
    E#L-X  = West  → Intersection   (ego spawn)
    E#D-X  = South → Intersection   (background spawn)
    E#X-T  = Intersection → North   (ego & background destination)
    E#X-R  = Intersection → East    (optional ego destination)
"""

from flow.networks import Network


ADDITIONAL_NET_PARAMS = {
    "length": 100,
    "num_lanes": 1,
    "speed_limit": 13.89,   # ≈ 50 km/h
}


class DeterministicSouthNorthNetwork(Network):
    """
    Deterministic 4-way intersection network for velocity-profile analysis.

    Ego routes (W→N and W→E) are specified with equal probability so that
    SUMO's route distribution mechanism works, but the environment subclass
    (alpha_env_deterministic.py) overrides the ego's actual route at episode
    start to make it fully deterministic.

    Background vehicles follow a single fixed route: S → N.
    """

    # ------------------------------------------------------------------
    # Route specification
    # ------------------------------------------------------------------
    def specify_routes(self, net_params):
        """
        Returns a route dict compatible with Flow's Network API.

        Each entry maps an *entering edge* to a list of (route, probability)
        tuples.  We use equal probabilities for the two ego routes so SUMO
        initialises without errors; the env overrides the chosen route later.
        """
        rts = {
            # ── Ego: enters from West ──────────────────────────────────
            # Route A: West → North  (straight / slight left depending on network)
            # Route B: West → East   (U-turn or right — crosses S→N stream)
            "E#L-X": [
                (["E#L-X", "E#X-T"], 0.5),   # W → N
                (["E#L-X", "E#X-R"], 0.5),   # W → E
            ],

            # ── Background: enters from South, always goes North ───────
            "E#D-X": [
                (["E#D-X", "E#X-T"], 1.0),   # S → N  (100 %)
            ],

            # ── Dummy entries for N and E inlets (no vehicles, but SUMO
            #    needs routes defined for every edge in the network) ─────
            "E#T-X": [
                (["E#T-X", "E#X-D"], 1.0),
            ],
            "E#R-X": [
                (["E#R-X", "E#X-L"], 1.0),
            ],
        }
        return rts
