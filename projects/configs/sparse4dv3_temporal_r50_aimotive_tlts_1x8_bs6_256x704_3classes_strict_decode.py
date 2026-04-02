from sparse4dv3_temporal_r50_aimotive_tlts_1x8_bs6_256x704_3classes import *

# Stricter decode setting for precision-oriented evaluation.
model["head"]["decoder"].update(
    dict(
        num_output=80,
        score_threshold=0.20,
    )
)
