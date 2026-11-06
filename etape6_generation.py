# etape6_generation.py
import os

from dotenv import load_dotenv
from openai import OpenAI

from etape5_recherche import rechercher

load_dotenv()  # lit le fichier .env

client = OpenAI(base_url=os.getenv("LLM_BASE_URL"), api_key=os.getenv("LLM_API_KEY"))
MODELE = os.getenv("LLM_MODEL")
REPLI = "Je ne trouve pas cette information dans la documentation."


def construire_prompt(question, extraits):
    contexte = ""
    for i, e in enumerate(extraits, start=1):
        contexte += f"[{i}] ({e['source']}, page {e['page']})\n{e['texte']}\n\n"
    return f"""Tu es l'assistant interne d'Altéa Services.
Réponds à la question en utilisant UNIQUEMENT les extraits ci-dessous.
Après chaque information, indique le numéro de l'extrait entre crochets, par exemple [1].
Si les extraits ne contiennent pas la réponse, réponds exactement : "{REPLI}"

Extraits :
{contexte}
Question : {question}
Réponse :"""


def repondre(question, k=3):
    extraits = rechercher(question, k)
    prompt = construire_prompt(question, extraits)
    reponse = client.chat.completions.create(
        model=MODELE,
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    return reponse.choices[0].message.content, extraits


if __name__ == "__main__":
    # Q4 : afficher le prompt complet
    q = "Combien de jours de télétravail par semaine sont autorisés chez Altéa ?"
    print(construire_prompt(q, rechercher(q)))
    print("=" * 60)

    # Q5 : 5 questions, dont 2 sans réponse dans la documentation
    questions = [
        "Combien de jours de télétravail par semaine sont autorisés chez Altéa ?",
        "Comment réinitialiser mon mot de passe ?",
        "Quel est le plafond de remboursement d'un repas d'affaires ?",
        "Quel temps fera-t-il demain ?",
        "Qui a gagné la Coupe du monde 2018 ?",
    ]
    for question in questions:
        reponse, extraits = repondre(question)
        print(f"\n### {question}\n{reponse}")
        print("  sources :", [f"{e['source']} p.{e['page']}" for e in extraits])

    # Q6 : même question posée au LLM SANS les extraits
    sans_rag = client.chat.completions.create(
        model=MODELE, temperature=0,
        messages=[{"role": "user", "content": q}],
    )
    print("\n### SANS RAG :", q)
    print(sans_rag.choices[0].message.content)