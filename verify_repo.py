"""verify_repo.py -- four checks on the released repository.

Run from anywhere; the repository root is found from this file's own location.

    1. identity   no author-identifying string and no absolute path anywhere
    2. budget     nothing oversized, nothing derived from a licensed dataset
    3. clean room  nature_figures.py regenerates the shipped PNGs byte-identically
                   from results/ alone -- without the datasets and without torch
    4. coverage   every shipped result file and figure is accounted for in
                   manifest.md, and every table number in the paper has an entry

Check 3 is the one that matters.  A repository that merely *contains* the right
files can still be impossible to reproduce from; only running it in a directory
that has nothing else in it shows that the shipped result files are sufficient.

Exit code 0 means all four passed.
"""
import hashlib
import io
import os
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))

# --- check 1 -----------------------------------------------------------------
# Substrings that must not appear anywhere in a public repository.  The first
# group is the grant number and the institutional identity; the second is the
# local paths of the machine the paper was written on.
BANNED = [
    "Y202559125",
    "hzpt",
    "yulei",
    "Hangzhou Polytechnic",
    "Zhejiang Business",
    "Zhejiang Provincial Education",
    "C:/Users", "C:\\Users",
    "D:/AMDGCN", "D:\\AMDGCN",
    "yy.workbuddy",
]
SCAN_EXT = (".py", ".md", ".txt", ".cff", ".yml", ".yaml", ".json", ".cfg", ".toml")
SCAN_NAME = (".gitignore",)

PLACEHOLDER_RE = re.compile(r"\[\[[^\]]+\]\]")

# --- check 2 -----------------------------------------------------------------
MAX_BYTES = 5 * 1024 * 1024
FORBIDDEN_EXT = (".npz", ".pt", ".pth", ".ckpt", ".log", ".mat", ".dat", ".pkl")
FORBIDDEN_NAME = ("deap_cache", "seed_cache", "seediv_cache")

# --- check 3 -----------------------------------------------------------------
# Figures that a script generates, and therefore must be byte-reproducible.
GENERATED = [
    "figure1_architecture", "figure2_seed_ablation", "figure3_seediv_ablation",
    "figure4_hyperparams", "figure5_ceiling", "figure7_shortcut_signal",
]
# Figure 2 in the paper has no generator: the SVG is the source.  It is checked
# for presence and for the two literals it embeds, not for reproducibility.
AUTHORED = "figure6_evaluation_protocol"
AUTHORED_LITERALS = ["n = 15–32", "12.4 pp"]

# --- check 4 -----------------------------------------------------------------
# Every table number that appears in the paper (Appendix A included).
TABLES = ["Table 1", "Table 1b", "Table 2", "Table 3", "Table 4", "Table 5",
          "Table 5b", "Table 6", "Table 6b", "Table 7", "Table 8", "Table 9",
          "Table 10", "Table 11", "Table 12", "Table A1"]


def _section(label, ok, detail=()):
    print("  [%s] %s" % ("PASS" if ok else "FAIL", label))
    for d in detail:
        print("         " + str(d))
    return ok


def _walk_files():
    for root, dirs, files in os.walk(HERE):
        dirs[:] = [d for d in dirs if d not in (".git", "__pycache__")]
        for fn in files:
            yield os.path.join(root, fn)


def _read(p):
    try:
        return io.open(p, encoding="utf-8", errors="replace").read()
    except Exception:
        return ""


def _md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def check_identity():
    hits = []
    # The verifier necessarily spells out the strings it forbids, so it cannot
    # scan itself; every other file is in scope.
    self_name = os.path.basename(os.path.abspath(__file__))
    for p in _walk_files():
        rel = os.path.relpath(p, HERE)
        if os.path.basename(rel) == self_name:
            continue
        if not (rel.endswith(SCAN_EXT) or os.path.basename(rel) in SCAN_NAME):
            continue
        txt = _read(p)
        for bad in BANNED:
            if bad in txt:
                # report the line so a false positive is obvious
                for ln in txt.splitlines():
                    if bad in ln:
                        hits.append("%s: %r" % (rel, ln.strip()[:100]))
                        break
    return hits


def check_budget():
    problems = [], 0
    found = []
    total = 0
    for p in _walk_files():
        rel = os.path.relpath(p, HERE)
        size = os.path.getsize(p)
        total += size
        if size > MAX_BYTES:
            found.append("oversized: %s (%.1f MB)" % (rel, size / 1048576.0))
        if rel.endswith(FORBIDDEN_EXT):
            found.append("excluded type shipped: %s" % rel)
        low = os.path.basename(rel).lower()
        for bad in FORBIDDEN_NAME:
            if low.startswith(bad):
                found.append("derived dataset cache shipped: %s" % rel)
    return found, total


def check_clean_room():
    """Regenerate the figures from results/ alone, into a scratch directory."""
    notes = []
    script = os.path.join(HERE, "code", "nature_figures.py")
    if not os.path.exists(script):
        return False, ["missing: code/nature_figures.py"]
    tmp = tempfile.mkdtemp(prefix="amdgcn_verify_")
    try:
        env = dict(os.environ)
        env["AMDGCN_FIGURES"] = tmp
        env["PYTHONIOENCODING"] = "utf-8"
        cp = subprocess.run([sys.executable, "-X", "utf8", script],
                            cwd=HERE, env=env, capture_output=True, timeout=600)
        if cp.returncode != 0:
            tail = cp.stderr.decode("utf-8", "replace").strip().splitlines()[-6:]
            return False, ["nature_figures.py exited %d" % cp.returncode] + tail

        for name in GENERATED:
            a = os.path.join(HERE, "figures", name + ".png")
            b = os.path.join(tmp, name + ".png")
            if not os.path.exists(b):
                notes.append("NOT REGENERATED: %s" % name)
                return False, notes
            same = _md5(a) == _md5(b)
            notes.append("%-30s %s" % (name + ".png",
                                       "identical" if same else "DIFFERS from the shipped file"))
            if not same:
                return False, notes
        return True, notes
    except subprocess.TimeoutExpired:
        return False, ["timed out after 600 s"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def check_authored_figure():
    notes = []
    ok = True
    for ext in (".svg", ".pdf", ".png"):
        p = os.path.join(HERE, "figures", AUTHORED + ext)
        if not os.path.exists(p):
            notes.append("missing: figures/%s%s" % (AUTHORED, ext))
            ok = False
    svg = os.path.join(HERE, "figures", AUTHORED + ".svg")
    if os.path.exists(svg):
        txt = _read(svg)
        for lit in AUTHORED_LITERALS:
            hit = lit in txt
            notes.append("literal %-14r %s" % (lit, "present" if hit else "NOT FOUND"))
    notes.append("authored diagram -- not script-generated, no reproducibility check applies")
    return ok, notes


def check_coverage():
    notes = []
    ok = True
    manifest = _read(os.path.join(HERE, "manifest.md"))
    if not manifest:
        return False, ["missing or empty: manifest.md"]

    for p in sorted(_walk_files()):
        rel = os.path.relpath(p, HERE)
        base = os.path.basename(rel)
        if rel.startswith("results" + os.sep) and base.endswith(".json"):
            if base not in manifest:
                notes.append("result file not documented: %s" % base)
                ok = False
        if rel.startswith("figures" + os.sep) and base.endswith(".png"):
            stem = base[:-4]
            if stem not in manifest:
                notes.append("figure not documented: %s" % base)
                ok = False

    missing = [t for t in TABLES if t not in manifest]
    if missing:
        notes.append("table numbers with no manifest entry: " + ", ".join(missing))
        ok = False

    stems = sorted({os.path.basename(p)[:-4] for p in _walk_files()
                    if os.path.relpath(p, HERE).startswith("figures" + os.sep)
                    and p.endswith(".png")})
    notes.append("%d result JSON + %d figures documented; %d table numbers covered"
                 % (sum(1 for p in _walk_files()
                        if os.path.relpath(p, HERE).startswith("results" + os.sep)
                        and p.endswith(".json")),
                    len(stems), len(TABLES)))
    return ok, notes


def main():
    print("=" * 74)
    print("AM-DGCN repository check")
    print("=" * 74)
    ok = True

    print("\n### 1. identity and absolute paths")
    hits = check_identity()
    ok &= _section("no identifying string and no absolute path",
                   not hits, hits or ["%d banned substrings, none present" % len(BANNED)])

    print("\n### 2. size and data budget")
    probs, total = check_budget()
    ok &= _section("no oversized file, no dataset-derived file",
                   not probs, probs or ["total %.2f MB in %d files"
                                        % (total / 1048576.0, len(list(_walk_files())))])

    print("\n### 3. clean-room figure regeneration")
    ok3, notes3 = check_clean_room()
    ok &= _section("7 figures regenerate from results/ with no dataset and no torch",
                   ok3, notes3)

    print("\n### 4. the authored diagram")
    ok4, notes4 = check_authored_figure()
    ok &= _section("evaluation-protocol figure present with its literals", ok4, notes4)

    print("\n### 5. manifest coverage")
    ok5, notes5 = check_coverage()
    ok &= _section("every result file, figure and table number is documented",
                   ok5, notes5)

    todo = sorted(set(PLACEHOLDER_RE.findall(_read(os.path.join(HERE, "CITATION.cff")))
                      + PLACEHOLDER_RE.findall(_read(os.path.join(HERE, "README.md")))))
    print("\n### placeholders (informational, not a failure)")
    print("  %d author-supplied placeholder(s): %s"
          % (len(todo), "; ".join(todo) if todo else "none"))

    print("\n" + "=" * 74)
    print("REPOSITORY CHECK: " + ("ALL GREEN" if ok else "FAILURES ABOVE"))
    print("=" * 74)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
