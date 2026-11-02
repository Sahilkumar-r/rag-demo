"""Download the FastAPI tutorial docs (MIT licensed) into data/docs/ with a manifest of source URLs."""
import json
import re
import urllib.request
from pathlib import Path

API = "https://api.github.com/repos/fastapi/fastapi/contents/docs/en/docs/tutorial"
RAW = "https://raw.githubusercontent.com/fastapi/fastapi/master/docs/en/docs/tutorial/{name}"
SITE = "https://fastapi.tiangolo.com/tutorial/{slug}/"
OUT = Path(__file__).resolve().parent.parent / "data" / "docs"


def get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "rag-demo"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    listing = json.loads(get(API))
    manifest = {}
    for item in listing:
        name = item["name"]
        if item["type"] != "file" or not name.endswith(".md"):
            continue
        text = get(RAW.format(name=name)).decode("utf-8")
        (OUT / name).write_text(text, encoding="utf-8")
        slug = re.sub(r"\.md$", "", name)
        manifest[name] = {"url": SITE.format(slug="" if slug == "index" else slug).replace("//", "/").replace(":/", "://")}
    (OUT.parent / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"downloaded {len(manifest)} docs to {OUT}")


if __name__ == "__main__":
    main()
