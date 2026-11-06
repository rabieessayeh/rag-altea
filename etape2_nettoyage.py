# etape2_nettoyage.py
import re

from etape1_collecte import collecter


def nettoyer(texte):
    texte = re.sub(r"[ \t]+", " ", texte)       # plusieurs espaces -> un seul
    texte = re.sub(r"\n{3,}", "\n\n", texte)    # 3 sauts de ligne ou plus -> 2
    texte = re.sub(r"-\n(\w)", r"\1", texte)    # recolle les mots coupés : "télé-\ntravail" -> "télétravail"
    return texte.strip()


def nettoyer_pages(pages):
    propres = []
    for p in pages:
        texte = nettoyer(p["texte"])
        if texte:                                # Q3 : on supprime les pages vides
            propres.append({**p, "texte": texte})
    return propres


if __name__ == "__main__":
    pages = collecter()
    propres = nettoyer_pages(pages)
    print(f"{len(pages)} pages avant, {len(propres)} après nettoyage\n")

    # Q4 : une même page avant / après
    exemple = next(p for p in pages
                   if p["source"] == "teletravail.pdf" and p["page"] == 2)
    print("===== AVANT =====")
    print(exemple["texte"][:300])
    print("\n===== APRÈS =====")
    print(nettoyer(exemple["texte"])[:300])