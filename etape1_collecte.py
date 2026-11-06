# etape1_collecte.py
from collections import Counter
from pathlib import Path

from pypdf import PdfReader


def lire_pdf(chemin):
    lecteur = PdfReader(chemin)
    pages = []
    for numero, page in enumerate(lecteur.pages, start=1):
        texte = page.extract_text() or ""
        pages.append({"source": chemin.name, "page": numero, "texte": texte})
    return pages


def lire_txt(chemin):
    texte = chemin.read_text(encoding="utf-8")
    return [{"source": chemin.name, "page": 1, "texte": texte}]


def collecter(dossier="docs"):
    pages = []
    for chemin in sorted(Path(dossier).rglob("*")):
        if not chemin.is_file():
            continue
        service = chemin.parent.name          # RH, IT ou General
        extension = chemin.suffix.lower()
        if extension == ".pdf":
            pages_doc = lire_pdf(chemin)
        elif extension == ".txt":
            pages_doc = lire_txt(chemin)
        else:
            continue
        for p in pages_doc:
            p["service"] = service
            pages.append(p)
    return pages


if __name__ == "__main__":
    pages = collecter()
    print(len(pages), "pages lues\n")

    # Q1 : nombre de pages par document
    compte = Counter((p["service"], p["source"]) for p in pages)
    for (service, source), n in sorted(compte.items()):
        print(f"{service:<8} {source:<26} {n} page(s)")