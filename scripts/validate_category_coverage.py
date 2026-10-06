"""Fail closed on distinct processed category coverage for final study views."""
from collections import Counter
CATEGORIES = ("charged", "mixed", "ccbar", "uubar", "ddbar", "ssbar")
def validate_coverage(records, *, quota=2000, diagnostic=False):
    seen=set();counts={cat:Counter(requested=0, attempted=0, processed=0, failed=0, truth_unavailable=0) for cat in CATEGORIES}
    for row in records:
        uid=row["event_uid"];cat=row["source_category"]
        if any(type(row[k]) is not bool for k in ("attempted","processed")): raise ValueError("Coverage flags must be boolean")
        if uid in seen: raise ValueError("Duplicate collision identity")
        if cat not in counts: raise ValueError("Unauthenticated category")
        seen.add(uid);c=counts[cat];c["requested"]+=1
        c["attempted"]+=int(row["attempted"])
        c["processed"]+=int(row["processed"])
        c["failed"]+=int(row["attempted"] and not row["processed"])
        c["truth_unavailable"]+=int(row.get("truth_unavailable",False))
        if row["processed"] and not row["attempted"]: raise ValueError("Processed without attempt")
    complete=all(counts[cat]["processed"]>=quota for cat in CATEGORIES)
    if not complete and not diagnostic: raise ValueError("Incomplete category coverage")
    return {"complete":complete,"diagnostic":diagnostic,"by_category":{k:dict(v) for k,v in counts.items()}}
