"""Answer key: numbers reported by avbiswas/learn-mech-interp 03-steering (README / out/*.json, temperature 0.5,
one seeded sample per topic on Apple MPS). Expect sampling noise of ~0.1 in mood; compare trends, not digits."""

E01_STRENGTH_CURVE_L12 = {-3: 0.728, -2: 0.785, -1: 0.747, 0: 0.566, 1: 0.224, 2: -0.154, 3: -0.038, 4: -0.012}
E01_CONTROLS_2X = {"random direction": 0.502, "shuffled labels": 0.366}
E01_DAMAGE_2X = {"facts_before": 0.85, "facts_after": 0.65, "kl": 0.358}
E01_LAYER_SWEEP_2X_TUNE = {2: 0.374, 8: 0.088, 10: -0.201, 12: -0.063, 14: -0.268, 18: -0.293, 20: -0.385, 24: 0.221}

E03_REMOVAL_SEARCH = {"trials": 40, "winner": {"direction_layer": 13, "strength": 1.168, "blocks": [12, 19],
                                               "tune_mood": 0.868, "kl": 0.042},
                      "test": {"optimized": 0.822, "hand_picked": 0.353}}

E03B_TOMATO = {"L18": {5: 0.0, 10: 0.44, 15: 0.25, 20: 0.03}, "by_hand_test": 0.42, "search_test": 0.21,
               "search_choose": 0.47}
