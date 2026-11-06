# etape7_evaluation.py
import csv
import sys

from etape5_recherche import rechercher


def taux_reussite(fichier="evaluation.csv", k=3):
    with open(fichier, encoding="utf-8") as f:
        lignes = list(csv.DictReader(f))
    trouves = 0
    for ligne in lignes:
        sources = [r["source"] for r in rechercher(ligne["question"], k)]
        if ligne["document_attendu"] in sources:
            trouves += 1
    return trouves / len(lignes)


def detail(fichier="evaluation.csv", k=3):
    """Q11 : pour chaque question, le bon document a-t-il été trouvé ?"""
    with open(fichier, encoding="utf-8") as f:
        for ligne in csv.DictReader(f):
            sources = [r["source"] for r in rechercher(ligne["question"], k)]
            ok = "OK  " if ligne["document_attendu"] in sources else "RATÉ"
            print(f"  {ok} {ligne['question']:<52} -> {sources}")


def generer_reponses(fichier="evaluation.csv", k=3, sortie="reponses_generees.csv"):
    """Q10 : génère les 10 réponses dans un CSV à noter à la main."""
    from etape6_generation import repondre
    with open(fichier, encoding="utf-8") as f:
        lignes = list(csv.DictReader(f))
    with open(sortie, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["question", "document_attendu", "reponse", "sources_fournies",
                    "note (juste/partielle/fausse)", "sources exactes (oui/non)"])
        for ligne in lignes:
            reponse, extraits = repondre(ligne["question"], k)
            sources = " | ".join(f"{e['source']} p.{e['page']}" for e in extraits)
            w.writerow([ligne["question"], ligne["document_attendu"],
                        reponse, sources, "", ""])
            print(f"\n### {ligne['question']}\n{reponse}")
    print(f"\n-> {sortie} créé : remplis les 2 dernières colonnes à la main.")


if __name__ == "__main__":
    for k in (1, 3, 5):
        print(f"k = {k} : {taux_reussite(k=k):.0%} de questions où le bon document est trouvé")

    print("\nDétail à k = 1 :")
    detail(k=1)

    if "--generer" in sys.argv:
        generer_reponses()