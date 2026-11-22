"""External literature sources (PubTator3, PubMed E-utilities, OpenAlex) with disk cache.

No user-identifying data is sent. Optional credentials come only from environment
variables the user sets themselves: OPENALEX_API_KEY, NCBI_API_KEY, HAKKEN_MAILTO.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

from . import __version__
from .common import log
from .workspace import cache_dir

UA = "hakken-discovery/%s (agent skill; python-urllib)" % __version__
_last_call = {}


def _throttle(host, min_interval):
    t = _last_call.get(host, 0.0)
    wait = min_interval - (time.time() - t)
    if wait > 0:
        time.sleep(wait)
    _last_call[host] = time.time()


def http_get(url, kind="json", ttl_days=30, min_interval=0.4, retries=3):
    key = hashlib.sha256(url.encode()).hexdigest()[:32]
    path = os.path.join(cache_dir("http"), key + (".json" if kind == "json" else ".txt"))
    if os.path.exists(path) and (time.time() - os.path.getmtime(path)) < ttl_days * 86400:
        with open(path, encoding="utf-8") as fh:
            data = fh.read()
        return json.loads(data) if kind == "json" else data
    host = urllib.parse.urlparse(url).netloc
    last = None
    for attempt in range(retries):
        _throttle(host, min_interval)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = resp.read().decode("utf-8", errors="replace")
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                fh.write(data)
            os.replace(tmp, path)
            return json.loads(data) if kind == "json" else data
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            last = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError("request failed after %d attempts: %s (%s)" % (retries, url.split("?")[0], last))


# --------------------------------------------------------------------------- #
# PubTator3: per-document relations with publication dates -> temporal triples
# --------------------------------------------------------------------------- #
PT = "https://www.ncbi.nlm.nih.gov/research/pubtator3-api"


def pubtator_search(query, max_docs):
    pmids, page = [], 1
    while len(pmids) < max_docs:
        url = "%s/search/?%s" % (PT, urllib.parse.urlencode({"text": query, "page": page}))
        d = http_get(url, min_interval=0.35)
        res = d.get("results") or []
        if not res:
            break
        pmids += [str(r["pmid"]) for r in res if r.get("pmid")]
        if page >= int(d.get("total_pages") or 0):
            break
        page += 1
    return pmids[:max_docs]


def _pt_name(role):
    acc = role.get("accession") or ""
    name = role.get("name") or ""
    if role.get("type") == "Gene" or not name:
        # accession like @GENE_TP53 -> TP53 ; keeps symbols canonical
        m = re.match(r"@[A-Z]+_(.+)", acc)
        if m:
            name = m.group(1).replace("_", " ") if role.get("type") != "Gene" else m.group(1)
    return name.strip(), (role.get("type") or "").lower()


def pubtator_relations(pmids, batch=100):
    """Yield ledger rows from PubTator3 BioC JSON exports."""
    for i in range(0, len(pmids), batch):
        chunk = pmids[i:i + batch]
        url = "%s/publications/export/biocjson?pmids=%s" % (PT, ",".join(chunk))
        d = http_get(url, min_interval=0.35, ttl_days=365)
        docs = d.get("PubTator3", d) if isinstance(d, dict) else d
        for doc in docs or []:
            year = (doc.get("date") or "")[:4]
            if not year.isdigit():
                continue
            rels = list(doc.get("relations") or [])
            for p in doc.get("passages") or []:
                rels += p.get("relations") or []
            for rel in rels:
                inf = rel.get("infons") or {}
                r1, r2 = inf.get("role1") or {}, inf.get("role2") or {}
                s, st = _pt_name(r1)
                o, ot = _pt_name(r2)
                rtype = inf.get("type")
                if not s or not o or not rtype or s == o:
                    continue
                yield {"subject": s, "relation": rtype, "object": o, "year": int(year),
                       "source": "PMID:%s" % doc.get("pmid") or doc.get("id"),
                       "evidence": "pubtator3 score=%s" % inf.get("score", ""),
                       "subject_type": st, "object_type": ot, "extractor": "pubtator3"}
        log("[pubtator] exported %d/%d documents" % (min(i + batch, len(pmids)), len(pmids)))


# --------------------------------------------------------------------------- #
# Evidence text for the language branch (single entity, published before a year)
# --------------------------------------------------------------------------- #
EU = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


def _ncbi_params(extra):
    p = dict(extra)
    if os.environ.get("NCBI_API_KEY"):
        p["api_key"] = os.environ["NCBI_API_KEY"]
    p["tool"] = "hakken-discovery"
    return p


def pubmed_search(term, before=None, after=None, retmax=20):
    q = {"db": "pubmed", "term": term, "retmax": retmax, "retmode": "json", "sort": "relevance"}
    if before or after:
        q.update({"datetype": "pdat", "mindate": str(after or 1800), "maxdate": str(before or 3000)})
    d = http_get("%s/esearch.fcgi?%s" % (EU, urllib.parse.urlencode(_ncbi_params(q))), min_interval=0.4)
    r = d.get("esearchresult", {})
    return int(r.get("count", 0)), r.get("idlist", [])


def pubmed_abstracts(pmids):
    if not pmids:
        return []
    q = {"db": "pubmed", "id": ",".join(pmids), "retmode": "xml", "rettype": "abstract"}
    xml = http_get("%s/efetch.fcgi?%s" % (EU, urllib.parse.urlencode(_ncbi_params(q))), kind="text",
                   min_interval=0.4, ttl_days=365)
    out = []
    root = ET.fromstring(xml)
    for art in root.iter("PubmedArticle"):
        pmid = art.findtext(".//PMID")
        year = art.findtext(".//PubDate/Year") or (art.findtext(".//PubDate/MedlineDate") or "")[:4]
        title = "".join(art.find(".//ArticleTitle").itertext()) if art.find(".//ArticleTitle") is not None else ""
        abst = " ".join("".join(a.itertext()) for a in art.findall(".//AbstractText"))
        out.append({"id": "PMID:%s" % pmid, "year": int(year) if str(year).isdigit() else None,
                    "title": title, "text": abst})
    return out


OA = "https://api.openalex.org"


def _oa_params(extra):
    p = dict(extra)
    if os.environ.get("OPENALEX_API_KEY"):
        p["api_key"] = os.environ["OPENALEX_API_KEY"]
    if os.environ.get("HAKKEN_MAILTO"):
        p["mailto"] = os.environ["HAKKEN_MAILTO"]
    return p


def openalex_search(query, before=None, after=None, per_page=20):
    filt = []
    if before:
        filt.append("to_publication_date:%s-12-31" % before)
    if after:
        filt.append("from_publication_date:%s-01-01" % after)
    q = {"search": query, "per-page": per_page,
         "select": "id,publication_year,title,abstract_inverted_index"}
    if filt:
        q["filter"] = ",".join(filt)
    d = http_get("%s/works?%s" % (OA, urllib.parse.urlencode(_oa_params(q))), min_interval=0.2)
    out = []
    for w in d.get("results", []):
        inv = w.get("abstract_inverted_index") or {}
        pos = sorted((i, word) for word, idx in inv.items() for i in idx)
        out.append({"id": w.get("id", "").replace("https://openalex.org/", "openalex:"),
                    "year": w.get("publication_year"), "title": w.get("title") or "",
                    "text": " ".join(word for _, word in pos)})
    return int((d.get("meta") or {}).get("count", 0)), out


_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(])")


def mention_sentences(docs, entity, exclude_terms=(), max_sentences=20):
    pat = re.compile(r"(?<![A-Za-z0-9])%s(?![A-Za-z0-9])" % re.escape(entity), re.I)
    excl = [re.compile(r"(?<![A-Za-z0-9])%s(?![A-Za-z0-9])" % re.escape(t), re.I) for t in exclude_terms if t]
    out = []
    for d in docs:
        for sent in _SENT.split((d.get("title", "") + ". " + d.get("text", "")).strip()):
            if pat.search(sent) and not any(e.search(sent) for e in excl):
                out.append({"id": d["id"], "year": d.get("year"), "sentence": sent.strip()})
                if len(out) >= max_sentences:
                    return out
    return out
