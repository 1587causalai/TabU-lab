"""Standard-library raw-table validation (also embedded in portable bundles)."""

import math


def validate_table(payload):
    table = payload.get("raw", payload).get("table", payload)
    values = table["values"]
    n = table["n_units"]
    d = table["n_features"]
    if type(n) is not int or type(d) is not int or n < 1 or d < 1 or len(values) != n:
        raise ValueError("invalid table shape")
    if table.get("n_rows", n) != n:
        raise ValueError("inconsistent n_rows")
    kinds = table["column_types"]
    classes = table["n_classes"]
    if len(kinds) != d or len(classes) != d:
        raise ValueError("type/domain width mismatch")
    for kind, card in zip(kinds, classes):
        if kind not in (
            "numeric",
            "ordinal",
            "binary",
            "categorical",
            "high_cardinality",
        ):
            raise ValueError("unknown column type")
        if kind == "numeric":
            if card is not None:
                raise ValueError("numeric column declares classes")
        elif type(card) is not int or card < 2 or (kind == "binary" and card != 2):
            raise ValueError("invalid category domain")
    for row in values:
        if len(row) != d or any(
            type(x) not in (int, float) or not math.isfinite(x) for x in row
        ):
            raise ValueError("values must be finite rectangular numbers")
        for x, card in zip(row, classes):
            if card is not None and (x != int(x) or not 0 <= x < card):
                raise ValueError("value outside declared domain")
    for key in ("missing_mask", "query_mask"):
        grid = table[key]
        if len(grid) != n or any(
            len(row) != d or any(type(x) is not bool for x in row) for row in grid
        ):
            raise ValueError("mask must be boolean and match values shape")
    counts = {
        "n_missing": sum(map(sum, table["missing_mask"])),
        "n_query": sum(map(sum, table["query_mask"])),
        "n_query_and_missing": sum(
            m and q
            for mr, qr in zip(table["missing_mask"], table["query_mask"])
            for m, q in zip(mr, qr)
        ),
    }
    shapes = table["shapes"]
    for key in ("values", "missing_mask", "query_mask"):
        if shapes[key] != [n, d]:
            raise ValueError("shape metadata mismatch")
    for key, v in counts.items():
        if shapes[key] != v:
            raise ValueError("mask count metadata mismatch")
    if table["query_mode"] == "label_cell":
        col = table["query_column"]
        if type(col) is not int or not 0 <= col < d:
            raise ValueError("invalid query column")
        if any(
            q and j != col for row in table["query_mask"] for j, q in enumerate(row)
        ):
            raise ValueError("query outside label column")
    elif table["query_mode"] != "any_cell":
        raise ValueError("invalid query mode")
    return dict(rows=n, columns=d, **counts)
