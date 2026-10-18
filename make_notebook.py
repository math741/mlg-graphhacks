"""Builds MLG_A100.ipynb from the cells below: python make_notebook.py"""
import nbformat as nbf

REPO = "https://github.com/math741/mlg-graphhacks"
BRANCH = "main"

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = [
    md("""# MLG ZERO-1T ΣΩ: an evolutionary mind on an A100

**Graph Hacks (FalkorDB) · track: Agent Memory & Coordination**

MLG is not a model. It is a *cognitive substrate*: the hardware is its body, and its rule is
**"redesign intelligence until every transistor has a reason to work."**

Here the MLG wakes up on an A100 **without being told what it is**, measures its own body, and then
**evolves its own brain**. Children inherit their parents' trained weights (Lamarckian evolution):
a deeper child keeps every layer its parent learned and grows new ones that start as the identity,
so the mind grows across generations without forgetting. Everything it learns lives in
**one FalkorDB graph**, and every "why?" is a Cypher query over that graph.

| Act | What happens | What lands in FalkorDB |
|---|---|---|
| I. Awakening | measures its body: memory, bandwidth cliffs, number formats, watts | `Body`, `Experiment`, `Measurement`, `Trait`, `PROVEN_BY` |
| II. Feeding | takes in the MLG corpus, plus Portuguese Wikipedia if you allow it | `Dataset`, sealed `Suite` |
| III. Evolution | conceives, trains, kills, crowns and grows brains | `Genome`, `PARENT_OF`, `INHERITED_FROM`, `Brain`, `Evaluation`, `Decision` |
| IV. Trust | explains every decision and detects a forged memory | SHA-256 chained `Event` log |

**Runtime:** `Runtime → Change runtime type → A100 GPU`. Any GPU or a CPU works too, slower;
the MLG measures what it gets."""),
    md("## 0 · Setup\nThe code comes from the public repository, or from `mlg-graphhacks.zip` if you "
       "uploaded it to the Files panel. Embedded FalkorDB (FalkorDBLite) runs inside this runtime. "
       "To use FalkorDB Cloud or a Docker server instead, set `FALKORDB_HOST`, `FALKORDB_PORT`, "
       "`FALKORDB_USERNAME` and `FALKORDB_PASSWORD` before the connect cell."),
    code(f"""import glob, os, sys
REPO_URL = "{REPO}"   # the public repository
BRANCH = "{BRANCH}"
QUICK = os.environ.get("MLG_QUICK") == "1"   # tiny budgets, for a smoke test
if os.path.exists("mlgsub"):                  # opened from a checkout
    CODE = os.path.abspath(".")
else:                                         # Colab
    if not os.path.exists("mlg-graphhacks/mlgsub"):
        zips = sorted(glob.glob("mlg-graphhacks*.zip"), key=os.path.getmtime)
        if zips:                              # uploaded to the Files panel, any name Colab gave it
            !unzip -q -o "{{zips[-1]}}"
        elif "YOUR_USER" not in REPO_URL:
            !git clone --depth 1 -b {{BRANCH}} {{REPO_URL}} mlg-graphhacks
    CODE = os.path.abspath("mlg-graphhacks")
if not os.path.exists(os.path.join(CODE, "mlgsub")):
    raise SystemExit("MLG code not found. Upload mlg-graphhacks.zip with the folder icon on the left, "
                     "wait for the upload to finish, then run this cell again.")
ROOT = os.path.join(CODE, "data")             # MLG corpus, dev suite and sealed Q root
!pip -q install falkordblite falkordb nvidia-ml-py
!ldd --version 2>/dev/null | head -1; nvidia-smi --query-gpu=name,memory.total,power.limit --format=csv || true
sys.path.insert(0, CODE)
print("code:", CODE, "| python", sys.version.split()[0])"""),
    code("""from mlgsub import connect, brain, explain, viz
from mlgsub.body import awaken
from mlgsub.evolve import evolve

try:
    sub = connect(graph="mlg", fresh=True)
except Exception as e:
    import traceback
    traceback.print_exc()
    raise SystemExit(f"FalkorDB unavailable: {type(e).__name__}: {e}\\n"
                     "Copy the error above, or set FALKORDB_HOST/PORT/USERNAME/PASSWORD "
                     "for FalkorDB Cloud and rerun this cell.")
print("substrate:", sub.mode, "|", sub.q("RETURN 'FalkorDB answers'")[0][0])"""),
    md("""## Act I · Awakening without assumptions
The MLG asks the driver what it is (**REPORTED**), then runs experiments on itself (**MEASURED**).
When bandwidth falls off a cliff it **bisects** that interval instead of sampling blindly
(manifesto XVIII). Each trait links to the measurements that prove it."""),
    code("""body = awaken(sub, quick=QUICK)
print()
print("\\n".join(explain.body_report(sub, body.id)))"""),
    code("""# Watch the body's self-knowledge being born, replayed from FalkorDB
viz.mind(sub, seconds=20, height=560)"""),
    code("""# Ask the graph: what proves the cache cliff, and how was it found?
for row in sub.q('''
    MATCH (:Body {id:$b})-[:HAS_TRAIT]->(t:Trait {name:'cache_cliff'})-[:PROVEN_BY]->(m:Measurement)
    RETURN m.working_set, m.value ORDER BY m.working_set''', b=body.id):
    print(f"working set {row[0] / 2**20:8.2f} MiB -> {row[1]:8.1f} GB/s")"""),
    md("""## Act II · Feeding
Big brains need more than the MLG's own 1.2 MB of text. Set `FEED_MB` to stream that many
megabytes of Portuguese Wikipedia (`wikimedia/wikipedia`, `20231101.pt`) into the training mix;
`0` keeps the MLG corpus only. The dev suite and the frozen Q root stay the MLG's own text, and any
of their lines found in the training mix is removed before a single step runs."""),
    code("""FEED_MB = 0 if QUICK else 256
extra = []
if FEED_MB:
    try:
        from datasets import load_dataset
    except ImportError:
        !pip -q install datasets
        from datasets import load_dataset
    path, n = "/content/wiki-pt.txt" if os.path.isdir("/content") else "wiki-pt.txt", 0
    if not os.path.exists(path) or os.path.getsize(path) < FEED_MB * 2**20:
        try:
            stream = load_dataset("wikimedia/wikipedia", "20231101.pt", split="train", streaming=True)
            with open(path, "w", encoding="utf-8") as fh:
                for row in stream:
                    fh.write(row["text"].strip() + "\\n\\n")
                    n += len(row["text"].encode())
                    if n >= FEED_MB * 2**20:
                        break
        except Exception as e:
            print(f"Wikipedia unavailable ({e}); feeding on the MLG corpus only.")
    if os.path.exists(path):
        extra = [path]
corpus = brain.Corpus(ROOT, body.device, extra=extra)
print(f"{len(corpus.train) / 2**20:,.1f} MiB of training text · {len(corpus.dev):,} dev bytes · "
      f"Q root sealed: {corpus.qroot_sealed} · {corpus.removed_lines} evaluation lines removed")"""),
    md("""## Act III · Evolution
Each genome describes a byte-level mixture-of-experts Transformer, the family of the MLG Resident
Core, with genes up to 1536 wide, 24 layers and 16 experts. The first candidate is the
**ancestor**: the architecture of the current Resident Core.

Every candidate goes through the furnace:
`SANDBOX` (predicted cost; killed without running if it cannot fit) → `TRAIN` → `ATTACK` (must beat
shuffled bytes) → `BENCHMARK` → `COMPARE`. Selection reads the **dev** suite; promotion needs the
**frozen Q root**. Most children **inherit** their parent's weights (`clone`, `grow`, `prune`,
`reshape`), so the best mind keeps learning from generation to generation. Prediction errors correct
the body's self-model, and the search dies on its own when it stops paying."""),
    code("""# Open the living mind in its own tab. It reads FalkorDB every two seconds, so you watch
# each genome, brain, KILL, COMMIT and golden thread of inheritance appear as the MLG writes it.
viz.live(sub, open_window=not QUICK)"""),
    code("""POPULATION, GENERATIONS, SECONDS = (3, 2, 20) if QUICK else (8, 6, 75)   # about an hour at most
furnace = evolve(sub, body, corpus, population=POPULATION, generations=GENERATIONS, seconds=SECONDS)"""),
    code("""viz.plot_evolution(furnace, f"Brains grown on {body.name}")
print("\\n".join(explain.why_champion(sub, body.id)))"""),
    code("""# The committed brain speaks (byte level; see the honesty box for what it is and is not)
ch = furnace.champion
for q in ["U: o que é a mlg\\nM:", "U: o que você faz quando a temperatura sobe?\\nM:"] if ch else []:
    print(brain.generate(ch["model"], ch["g"], q, body.device), "\\n")"""),
    code("""# Talk to the champion. Every COMMIT was saved to champions/*.pt, so it survives the session
# (download it from the Files panel; brain.load_brain(path) brings it back anywhere).
# It is a small byte-level brain trained for minutes: expect the MLG's voice, not a chatbot.
print("saved champions:", sorted(os.listdir("champions")) if os.path.isdir("champions") else [])
if ch and not QUICK:
    while True:
        q = input("you (empty to stop): ")
        if not q.strip():
            break
        print("MLG:", brain.reply(ch["model"], ch["g"], q, body.device), "\\n")"""),
    code("""# Inheritance in the graph: which brains carry whose mind?
for child, mode, parent, bpb in sub.q('''
    MATCH (g:Genome)-[r:INHERITED_FROM]->(p:Brain), (g)-[:EXPRESSES]->(b:Brain)
    RETURN b.id, r.mode, p.id, b.dev_bpb ORDER BY g.generation, b.id'''):
    print(f"{child[6:18]} <-{mode:^9s}- {parent[6:18]}   dev {(bpb or float('nan')):.3f} bpb")"""),
    code("""# The evolution, alive: genomes conceived, brains trained, golden threads of inheritance,
# red shockwaves for KILL, a golden burst and a crown for every COMMIT
viz.mind(sub, seconds=60, height=780)"""),
    code("""# Keep evolving from where it stopped: same champion, same parents, same inheritable
# weights, generations numbered after the last one. Watch it in the live tab.
furnace = evolve(sub, body, corpus, furnace=furnace, population=POPULATION,
                 generations=1 if QUICK else 4, seconds=SECONDS)
print("\\n".join(explain.why_champion(sub, body.id)))"""),
    md("""## Act IV · Trust: nothing promoted by faith
Every event is chained by SHA-256, like `core/store.c`. Forge one memory and the chain breaks
at exactly that point."""),
    code("""print("chain:", sub.verify_events())
orig = sub.q("MATCH (e:Event {seq:5}) RETURN e.detail")[0][0]
sub.q("MATCH (e:Event {seq:5}) SET e.detail = '{\\"forged\\": true}'")
print("after forging event 5:", sub.verify_events())
sub.q("MATCH (e:Event {seq:5}) SET e.detail = $d", d=orig)
print("restored:", sub.verify_events())"""),
    code("""# Ask anything. Why was each brain killed?
for gid, verdict, stage, reason in sub.q('''
    MATCH (d:Decision)-[:KILLED]->(x) WHERE d.verdict IN ['KILL','ROLLBACK']
    RETURN x.id, d.verdict, d.stage, d.reason ORDER BY d.id LIMIT 20'''):
    print(f"{verdict:8s} {stage:8s} {gid:34s} {reason}")
print(sub.q("MATCH (n:Object) RETURN n.type, count(*) ORDER BY count(*) DESC"))"""),
    md("""## Honesty box (manifesto LXIII: no fake numbers)
- **MEASURED** traits come from experiments in this runtime; **REPORTED** ones are driver claims;
  **INFERRED** ones are derived from measurements. Energy is NOT MEASURED where no sensor exists.
- Quality is bits per byte on held-out MLG text (dev for selection, the frozen Q root for promotion).
  Q root and dev lines found in the training mix are removed before training.
- The brains are small byte-level models trained for minutes. They model the MLG's text measurably
  better than chance and better each generation; they are not general chatbots.
- The C MLG (daemon, Resident Core, Q root, corpus) predates the hackathon. **New for Graph Hacks:**
  the FalkorDB substrate, zero-assumption body discovery, the brain-genome furnace with Lamarckian
  inheritance, the self-model, the living-mind view and graph explanations.
- Built with AI coding assistance (Claude), reviewed and directed by the author."""),
]

nb = nbf.v4.new_notebook(cells=cells, metadata={
    "accelerator": "GPU", "colab": {"gpuType": "A100", "provenance": []},
    "kernelspec": {"name": "python3", "display_name": "Python 3"},
    "language_info": {"name": "python"}})
nbf.write(nb, "MLG_A100.ipynb")
print("wrote MLG_A100.ipynb with", len(cells), "cells")
