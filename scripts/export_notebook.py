"""Regenerate scripts/m2m_pipeline.py (read-only percent-format export) from the canonical notebook."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
nb = json.loads((ROOT / "notebooks" / "M2M_Metadata_to_Model.ipynb").read_text())
out = ['"""Auto-generated, read-only export of notebooks/M2M_Metadata_to_Model.ipynb (percent format).\n\n'
       'The notebook is the canonical source; regenerate this file with `python scripts/export_notebook.py`.\n"""\n']
for c in nb["cells"]:
    s = "".join(c["source"]) if isinstance(c["source"], list) else c["source"]
    if c["cell_type"] == "markdown":
        out.append("# %% [markdown]\n" + "\n".join("# " + l if l else "#" for l in s.split("\n")) + "\n")
    else:
        out.append("# %%\n" + s + "\n")
(ROOT / "scripts" / "m2m_pipeline.py").write_text("\n".join(out))
print("wrote scripts/m2m_pipeline.py")
