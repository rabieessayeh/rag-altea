# etape5_recherche.py
import chromadb
from sentence_transformers import SentenceTransformer

modele = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
collection = chromadb.PersistentClient(path="base_vectorielle").get_collection("altea")


def rechercher(question, k=3, service=None):
    vecteur = modele.encode([question]).tolist()
    filtre = {"service": service} if service else None
    res = collection.query(query_embeddings=vecteur, n_results=k, where=filtre)
    resultats = []
    for texte, meta, distance in zip(res["documents"][0],
                                     res["metadatas"][0],
                                     res["distances"][0]):
        resultats.append({
            "texte": texte,
            "source": meta["source"],
            "page": meta["page"],
            "service": meta["service"],
            "score": 1 - distance,      # proche de 1 = très proche
        })
    return resultats


def afficher(question, k=3, service=None):
    print(f"\n### {question}  (k={k}, service={service})")
    for r in rechercher(question, k, service):
        print(f"  {r['score']:.3f}  [{r['service']}] {r['source']} p.{r['page']}")


if __name__ == "__main__":
    # Q1 : 3 questions
    afficher("Combien de jours de télétravail par semaine ?")
    afficher("Mon VPN ne marche pas, que faire ?")
    afficher("Quel est le plafond pour un repas du soir ?")

    # Q2 : influence de k
    for k in (1, 3, 10):
        afficher("Quel est le plafond pour un repas du soir ?", k=k)

    # Q3 : filtre par service
    afficher("Combien de jours de télétravail par semaine ?", service="RH")
    afficher("Combien de jours de télétravail par semaine ?", service="IT")