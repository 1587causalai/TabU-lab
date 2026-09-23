"""Read sparse W&B history without counting inclusive page boundaries twice."""


def scan_unique(run, keys):
    rows = {}
    for row in run.scan_history(keys=list(dict.fromkeys(["_step", *keys])), page_size=1000):
        step = row["_step"]
        if step in rows and rows[step] != row:
            raise ValueError("conflicting cloud payloads for one history step")
        rows[step] = row
    return [rows[step] for step in sorted(rows)]
