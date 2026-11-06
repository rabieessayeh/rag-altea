# etape4_indexation.py
import chromadb
from sentence_transformers import SentenceTransformer, util

from etape1_collecte import collecter
from etape2_nettoyage import nettoyer_pages
from etape3_decoupage import decouper_pages

modele = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# ---------- Partie A : découvrir les embeddings (Q8, Q9) ----------
phrases = [
    "Comment poser un jour de congé ?",
    "Quelle est la procédure pour demander des vacances ?",
    "Mon ordinateur ne se connecte pas au VPN.",
]
vecteurs = modele.encode(phrases)
print("Forme :", vecteurs.shape)                 # (3, dimension)
print(util.cos_sim(vecteurs, vecteurs))

# ---------- Partie B : indexer tout le corpus (Q10) ----------
pages = nettoyer_pages(collecter())
chunks = decouper_pages(pages, taille=500, chevauchement=50)
print(f"\n{len(chunks)} chunks produits")

client = chromadb.PersistentClient(path="base_vectorielle")
collection = client.get_or_create_collection(
    "altea", metadata={"hnsw:space": "cosine"})

textes = [c["texte"] for c in chunks]
vecteurs = modele.encode(textes, show_progress_bar=True).tolist()

collection.upsert(
    ids=[c["id"] for c in chunks],
    documents=textes,
    embeddings=vecteurs,
    metadatas=[{"source": c["source"], "page": c["page"], "service": c["service"]}
               for c in chunks],
)
print(collection.count(), "chunks indexés")
assert collection.count() == len(chunks)

# ---------- Q11 : premières recherches ----------
questions = [
    "Combien de jours de télétravail ?",
    "J'ai oublié mon mot de passe, que faire ?",
    "À quelle heure ouvre le site de Lyon ?",
]
for q in questions:
    resultat = collection.query(
        query_embeddings=modele.encode([q]).tolist(), n_results=3)
    print(f"\n### {q}")
    for doc, meta, dist in zip(resultat["documents"][0],
                               resultat["metadatas"][0],
                               resultat["distances"][0]):
        print(f"  {meta['source']} p.{meta['page']}  distance={dist:.3f}")