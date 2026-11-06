# etape3_decoupage.py
from etape1_collecte import collecter
from etape2_nettoyage import nettoyer_pages


def decouper(texte, taille=500, chevauchement=50):
    assert chevauchement < taille, "le chevauchement doit être plus petit que la taille"
    morceaux = []
    debut = 0
    while debut < len(texte):
        fin = debut + taille                  # fin du morceau
        morceaux.append(texte[debut:fin])
        if fin >= len(texte):
            break
        debut = fin - chevauchement           # on recule pour créer le chevauchement
    return morceaux


def decouper_pages(pages, taille=500, chevauchement=50):
    chunks = []
    for p in pages:
        for i, morceau in enumerate(decouper(p["texte"], taille, chevauchement)):
            chunks.append({
                "id": f"{p['source']}-p{p['page']}-{i}",
                "texte": morceau,
                "source": p["source"],
                "page": p["page"],
                "service": p["service"],
            })
    return chunks


if __name__ == "__main__":
    # Q5
    resultat = decouper("abcdefghij", taille=4, chevauchement=1)
    print("Q5 :", resultat)
    assert resultat == ["abcd", "defg", "ghij"]

    # Q6
    pages = nettoyer_pages(collecter())
    print(f"\n{'Taille':>7} | {'Chevauch.':>9} | {'Nb chunks':>9} | {'Long. moy.':>10}")
    for taille, chev in [(200, 20), (500, 50), (1000, 100)]:
        chunks = decouper_pages(pages, taille, chev)
        moyenne = sum(len(c["texte"]) for c in chunks) / len(chunks)
        print(f"{taille:>7} | {chev:>9} | {len(chunks):>9} | {moyenne:>10.0f}")