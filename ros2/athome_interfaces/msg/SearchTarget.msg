# One target being searched (part of SearchTargets).
string name                   # target as commanded (e.g. "lamp")
# Labels that count as finding it (the target category and its subtypes,
# e.g. "lamp", "desk lamp", "floor lamp"). Perception adds them to its
# detection queries and reports a detection with one of these labels.
string[] query_names
