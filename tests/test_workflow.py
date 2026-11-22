"""End-to-end smoke tests for the hakken-discovery skill (stdlib unittest, no network).

    python3 -m unittest discover -s tests -v
"""
import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKILL = os.path.join(ROOT, "skills", "hakken-discovery")
H = [sys.executable, os.path.join(SKILL, "scripts", "hakken.py")]
TOY = os.path.join(SKILL, "examples", "toy_triples.csv")

sys.dont_write_bytecode = True  # keep the skill directory clean
sys.path.insert(0, os.path.join(SKILL, "scripts"))
from hakkenlib.graph import apply_config  # noqa: E402


def run(args, cwd, ok=(0,)):
    p = subprocess.run(H + args, cwd=cwd, capture_output=True, text=True,
                       env=dict(os.environ, HAKKEN_CACHE=os.path.join(cwd, ".cache")))
    if p.returncode not in ok:
        raise AssertionError("hakken %s failed (%d):\n%s\n%s" % (args, p.returncode, p.stdout, p.stderr))
    return p


class Workflow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.cwd = cls.tmp.name
        run(["init", "--goal", "toy"], cls.cwd)
        run(["ingest", TOY, "--extractor", "synthetic"], cls.cwd)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def ws(self, *p):
        return os.path.join(self.cwd, "hakken", *p)

    def latest_run(self, kind="predict"):
        runs = sorted(d for d in os.listdir(self.ws("runs")) if "_%s_" % kind in d)
        return self.ws("runs", runs[-1])

    def test_01_workspace_layout(self):
        for f in ("hakken.json", "README.md", ".gitignore", "config/relations.json",
                  "config/entities.json", "ledger/observations.csv"):
            self.assertTrue(os.path.exists(self.ws(f)), f)
        with open(self.ws(".gitignore")) as fh:
            gi = fh.read()
        self.assertIn("runs/", gi)
        self.assertIn("judgments/", gi)  # private policy by default
        # re-ingesting the same file adds nothing (dedupe on s, r, o, year, source)
        p = run(["ingest", TOY, "--extractor", "synthetic"], self.cwd)
        self.assertIn(" 0 new observations", p.stderr)

    def test_02_backtest_beats_null(self):
        run(["backtest", "--cutoff", "2008", "--bins", "2"], self.cwd)
        res = json.load(open(os.path.join(self.latest_run("backtest"), "backtest.json")))
        for r in res["results"]:
            first = r["bins"][0]
            self.assertGreater(first["micro"], 3 * first["null_micro"], r["model"])

    def test_03_predict_select_explain_consistent(self):
        run(["predict", "--cutoff", "2015"], self.cwd)
        rd = self.latest_run()
        run(["select", "--top-k", "2", "--limit", "5"], self.cwd)
        rows = list(csv.DictReader(open(os.path.join(rd, "shortlist.csv"))))
        self.assertTrue(rows)
        top = rows[0]
        run(["explain", "-s", top["subject"], "-r", top["relation"], "-O", top["object"]], self.cwd)
        ex = json.load(open(os.path.join(rd, "explanations", "explanations.json")))[0]
        # explanation scorer must reproduce the prediction (orientation-independent model)
        self.assertAlmostEqual(ex["score"], float(top["prob"]), places=4)
        self.assertTrue(ex["explanations"])

    def test_04_language_branch_and_fuse(self):
        run(["predict", "--cutoff", "2015"], self.cwd)
        rd = self.latest_run()
        run(["judge-queue", "--top", "3", "--batch-size", "2"], self.cwd)
        items = [json.loads(l) for l in open(os.path.join(rd, "judge", "batch_001.jsonl"))]
        out = os.path.join(self.cwd, "j.jsonl")
        with open(out, "w") as fh:
            for it in items:
                fh.write(json.dumps(dict(it, score=0.9, model="test")) + "\n")
            fh.write(json.dumps({"subject": "a", "relation": "b", "object": "c", "cutoff": 2015, "score": 7}) + "\n")
        p = run(["record", "judgments", out], self.cwd)
        self.assertIn("1 invalid", p.stderr)
        p = run(["judge-queue", "--top", "3"], self.cwd)
        self.assertIn("2 already judged", p.stderr)
        run(["fuse"], self.cwd)
        fused = list(csv.DictReader(open(os.path.join(rd, "fused.csv"))))
        judged = [r for r in fused if r["judged"] == "1"]
        self.assertEqual(len(judged), 2)
        for r in judged:
            self.assertGreater(float(r["fused"]), float(r["graph_prob"]))

    def test_05_policy_gate(self):
        run(["predict", "--cutoff", "2015", "--limit", "50"], self.cwd)
        p = run(["novelty", "--top", "5", "--external", "pubmed"], self.cwd, ok=(3,))
        self.assertIn("--confirmed", p.stderr)
        p = run(["novelty", "--top", "5", "--mark-pending"], self.cwd)
        self.assertIn("pending", p.stdout)

    def test_06_report(self):
        run(["predict", "--cutoff", "2015"], self.cwd)
        run(["select", "--limit", "3"], self.cwd)
        run(["report"], self.cwd)
        txt = open(os.path.join(self.latest_run(), "report.md")).read()
        self.assertIn("### H1.", txt)
        self.assertIn("<!-- agent", txt)


class Cleaning(unittest.TestCase):
    def test_polarity_swap_synonyms(self):
        raw = [("TP53", "increases expression of", "BAMBI", 2001, {}),
               ("p53", "decreases expression of", "BAMBI", 1999, {}),
               ("BAMBI", "is treated by", "DrugX", 2003, {}),
               ("GeneA", "increases expression of", "GeneB", 2004, {}),
               ("GeneA", "increases expression of", "GeneA", 2004, {})]
        rel = {"swap": {"IS_TREATED_BY": "TREATS"}, "upper_snake": True,
               "polarity": [{"parent": "AFFECTS_EXPRESSION_OF",
                             "children": ["INCREASES_EXPRESSION_OF", "DECREASES_EXPRESSION_OF"]}]}
        first, support, _, stats = apply_config(raw, rel, {"synonyms": {"p53": "TP53"}}, verbose=False)
        self.assertEqual(first[("TP53", "AFFECTS_EXPRESSION_OF", "BAMBI")], 1999)
        self.assertNotIn(("TP53", "INCREASES_EXPRESSION_OF", "BAMBI"), first)
        self.assertIn(("DrugX", "TREATS", "BAMBI"), first)
        self.assertEqual(first[("GeneA", "AFFECTS_EXPRESSION_OF", "GeneB")], 2004)
        self.assertEqual(stats["self_loop"], 1)
        self.assertEqual(support[("TP53", "AFFECTS_EXPRESSION_OF", "BAMBI")], 2)


if __name__ == "__main__":
    unittest.main()
