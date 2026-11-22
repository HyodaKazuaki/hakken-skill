"""Shared I/O helpers: CSV/JSONL, atomic writes, locking, year parsing, relation tokens."""
from __future__ import annotations

import contextlib
import csv
import datetime as _dt
import json
import os
import re
import sys
import tempfile
import time

ALIASES = {
    "subject": ["subject", "s", "head", "src", "subject_name"],
    "relation": ["relation", "r", "predicate", "rel", "relation_type", "edge_type"],
    "object": ["object", "o", "tail", "dst", "object_name"],
    "year": ["year", "time", "date", "timestamp", "pub_year", "first_year"],
}

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))


def log(msg):
    print(msg, file=sys.stderr)


def die(msg, code=2):
    log("error: " + msg)
    sys.exit(code)


def now_iso():
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()


def parse_year(v):
    v = str(v).strip()
    if re.fullmatch(r"-?\d+", v):  # plain integer: year, yyyymm or any ordinal time step
        return int(v)
    m = re.search(r"(\d{4})", v)
    if not m:
        raise ValueError("cannot parse time from %r" % (v,))
    return int(m.group(1))


def sniff_delimiter(path):
    with open(path, encoding="utf-8-sig") as fh:
        head = fh.readline()
    return "\t" if head.count("\t") > head.count(",") else ","


def read_table(path):
    """Read CSV/TSV into list of dicts."""
    delim = sniff_delimiter(path)
    with open(path, newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh, delimiter=delim))


def read_triples(path):
    """Read a triple file -> list of (s, r, o, year, extra_dict). Column aliases accepted."""
    delim = sniff_delimiter(path)
    out, bad = [], 0
    with open(path, newline="", encoding="utf-8-sig") as fh:
        rd = csv.DictReader(fh, delimiter=delim)
        cols = {c.lower().strip(): c for c in (rd.fieldnames or [])}
        mapping = {}
        for key, names in ALIASES.items():
            for n in names:
                if n in cols:
                    mapping[key] = cols[n]
                    break
            if key not in mapping:
                die("column for %r not found in %s (have: %s)" % (key, path, ", ".join(rd.fieldnames or [])))
        used = set(mapping.values())
        for row in rd:
            try:
                s = (row[mapping["subject"]] or "").strip()
                r = (row[mapping["relation"]] or "").strip()
                o = (row[mapping["object"]] or "").strip()
                y = parse_year(row[mapping["year"]])
            except (ValueError, KeyError, TypeError):
                bad += 1
                continue
            if not s or not r or not o:
                bad += 1
                continue
            extra = {k: v for k, v in row.items() if k not in used and k is not None}
            out.append((s, r, o, y, extra))
    if bad:
        log("[read] skipped %d malformed rows in %s" % (bad, path))
    return out


def _open_out(path):
    if path in (None, "-"):
        return sys.stdout
    return open(path, "w", newline="", encoding="utf-8")


@contextlib.contextmanager
def atomic_open(path, mode="w"):
    """Write to a temp file in the same directory, then os.replace() into place."""
    d = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp-", dir=d)
    try:
        with os.fdopen(fd, mode, newline="" if "b" not in mode else None,
                       encoding=None if "b" in mode else "utf-8") as fh:
            yield fh
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def write_rows(path, fieldnames, rows):
    """Write dict rows as CSV (atomically when path is a file, '-' = stdout)."""
    if path in (None, "-"):
        w = csv.DictWriter(sys.stdout, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)
        return
    with atomic_open(path) as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)


def write_text(path, text):
    with atomic_open(path) as fh:
        fh.write(text)


def write_json(path, obj):
    with atomic_open(path) as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False)
        fh.write("\n")


def read_json(path, default=None):
    if not os.path.exists(path):
        return default
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def read_jsonl(path):
    out = []
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as fh:
        for i, ln in enumerate(fh, 1):
            ln = ln.strip()
            if not ln:
                continue
            try:
                out.append(json.loads(ln))
            except json.JSONDecodeError:
                log("[jsonl] %s:%d is not valid JSON - skipped" % (path, i))
    return out


def append_jsonl(path, records):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with file_lock(path + ".lock"):
        with open(path, "a", encoding="utf-8") as fh:
            for r in records:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")


@contextlib.contextmanager
def file_lock(lock_path, timeout=60.0, stale=600.0):
    """Cross-process advisory lock via O_EXCL lock file (portable, no fcntl needed)."""
    start = time.time()
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            break
        except FileExistsError:
            try:
                if time.time() - os.path.getmtime(lock_path) > stale:
                    os.unlink(lock_path)
                    continue
            except OSError:
                continue
            if time.time() - start > timeout:
                die("could not acquire lock %s (delete it if no other hakken process runs)" % lock_path)
            time.sleep(0.1)
    try:
        yield
    finally:
        with contextlib.suppress(OSError):
            os.unlink(lock_path)


def pkey(a, b):
    return (a, b) if a <= b else (b, a)


def inv(tok):
    """Invert a directed relation token ('R>' <-> 'R<')."""
    return tok[:-1] + ("<" if tok[-1] == ">" else ">")


def tok_to_triple(u, v, tok):
    rel = tok[:-1]
    return (u, rel, v) if tok[-1] == ">" else (v, rel, u)


def triple_to_tok(u, v, s, r, o):
    """Token for triple (s, r, o) seen from u where {u, v} == {s, o}."""
    return r + (">" if s == u else "<")


def read_list(path):
    if not path:
        return None
    with open(path, encoding="utf-8") as fh:
        return [ln.strip() for ln in fh if ln.strip() and not ln.lstrip().startswith("#")]


def hyp_key(s, r, o):
    return "%s|%s|%s" % (s, r, o)


def sigmoid(z):
    z = max(-35.0, min(35.0, z))
    import math
    return 1.0 / (1.0 + math.exp(-z))


def logit(p, eps=1e-6):
    import math
    p = min(1 - eps, max(eps, p))
    return math.log(p / (1 - p))
