"""Build BibTeX entries from Crossref metadata (publisher records) for a list of key<TAB>DOI lines.

Usage: python make_bib.py dois.tsv out.bib
Titles keep their published capitalisation; acronyms and mixed-case words are brace-protected.
"""
import html
import json
import re
import sys
import time
import unicodedata
import urllib.request

UA = {"User-Agent": "thesis-bibcheck/1.0 (python urllib)"}

VENUES = [  # (substring of container title, booktitle, short)
    ("Computer Vision and Pattern Recognition Workshops", "Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)"),
    ("Computer Vision and Pattern Recognition", "Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)"),
    ("International Conference on Computer Vision", "Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)"),
    ("Winter Conference on Applications of Computer Vision", "Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)"),
    ("Acoustics, Speech and Signal Processing", "Proceedings of the IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)"),
    ("Intelligent Robots and Systems", "Proceedings of the IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)"),
    ("European Signal Processing Conference", "Proceedings of the European Signal Processing Conference (EUSIPCO)"),
    ("Interspeech", "Proceedings of Interspeech"),
    ("Empirical Methods in Natural Language Processing: System Demonstrations", "Proceedings of the Conference on Empirical Methods in Natural Language Processing: System Demonstrations (EMNLP)"),
    ("International Conference on Multimodal Interaction", "Proceedings of the International Conference on Multimodal Interaction (ICMI)"),
    ("ACM International Conference on Multimedia", "Proceedings of the ACM International Conference on Multimedia (ACM MM)"),
    ("Computer Vision, Imaging and Computer Graphics Theory and Applications", "Proceedings of the International Joint Conference on Computer Vision, Imaging and Computer Graphics Theory and Applications (VISAPP)"),
    ("AAAI Conference on Artificial Intelligence", "Proceedings of the AAAI Conference on Artificial Intelligence"),
]


def get(url):
    for k in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as e:  # noqa: BLE001
            err = e
            time.sleep(3 + 4 * k)
    raise RuntimeError(f"{url}: {err}")


def tex(s: str) -> str:
    s = html.unescape(s or "")
    s = unicodedata.normalize("NFC", s).replace(" ", " ").replace(" ", " ")
    s = re.sub(r"<[^>]+>", "", s)  # MathML/JATS tags
    s = s.replace("&", r"\&").replace("%", r"\%").replace("#", r"\#").replace("_", r"\_")
    s = s.replace("–", "--").replace("—", "---").replace("‘", "`").replace("’", "'")
    s = s.replace("“", "``").replace("”", "''").replace("°", r"$^\circ$")
    return re.sub(r"\s+", " ", s).strip()


def protect(title: str) -> str:
    def rep(m):
        w = m.group(0)
        inner = w[1:]
        if sum(c.isupper() for c in w) >= 2 or any(c.isdigit() for c in w) or re.search(r"[A-Z]", inner):
            return "{" + w + "}"
        return w
    return re.sub(r"[A-Za-z][A-Za-z0-9\-]*[A-Za-z0-9]|[A-Z]", rep, title)


def year_of(m):
    for f in ("published-print", "published-online", "issued"):
        d = (m.get(f) or {}).get("date-parts")
        if d and d[0] and d[0][0]:
            return d[0][0]
    return None


def entry(key, doi):
    m = get("https://api.crossref.org/works/" + doi)["message"]
    typ = m.get("type")
    ct = [tex(c) for c in (m.get("container-title") or [])]
    authors = " and ".join(
        tex(f"{a.get('family', '')}, {a.get('given', '')}".strip(", ")) if a.get("family") else tex(a.get("name", ""))
        for a in m.get("author", []))
    f = {"author": authors, "title": "{" + protect(tex((m.get("title") or [""])[0])) + "}", "year": str(year_of(m))}
    if typ == "journal-article" and not any(v[0] in " ".join(ct) for v in VENUES):
        kind = "article"
        f["journal"] = ct[0] if ct else ""
        if m.get("volume"):
            f["volume"] = m["volume"]
        if m.get("issue"):
            f["number"] = m["issue"]
    else:
        kind = "inproceedings"
        bt = None
        for sub, name in VENUES:
            if any(sub in c for c in ct):
                bt = name
        if bt is None:  # Springer LNCS book chapters: book title is the second container title
            books = [c for c in ct if c != "Lecture Notes in Computer Science"]
            bt = books[0] if books else (ct[0] if ct else "")
            if "Lecture Notes in Computer Science" in ct:
                f["series"] = "Lecture Notes in Computer Science"
        f["booktitle"] = bt
        if m.get("volume") and "AAAI" in bt:
            f["volume"] = m["volume"]
    if m.get("page"):
        f["pages"] = m["page"].replace("-", "--")
    elif m.get("article-number"):
        f["eid"] = m["article-number"]
    if m.get("publisher") and kind == "inproceedings":
        f["publisher"] = tex(m["publisher"])
    f["doi"] = m["DOI"]
    body = ",\n".join(f"  {k:<9} = {{{v}}}" if k != "title" else f"  {k:<9} = {v}" for k, v in f.items())
    return f"@{kind}{{{key},\n{body}\n}}\n", m


def main():
    rows = [ln.split() for ln in open(sys.argv[1], encoding="utf-8") if ln.strip() and not ln.startswith("#")]
    out, meta = [], {}
    for key, doi in rows:
        e, m = entry(key, doi)
        out.append(e)
        meta[key] = {"doi": doi, "type": m.get("type"), "container": m.get("container-title")}
        print(e, flush=True)
        time.sleep(0.4)
    open(sys.argv[2], "w", encoding="utf-8", newline="\n").write("\n".join(out))
    json.dump(meta, open(sys.argv[2] + ".meta.json", "w"), indent=1)


if __name__ == "__main__":
    main()
