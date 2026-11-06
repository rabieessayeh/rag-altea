# app.py
import streamlit as st

from etape6_generation import repondre

st.title("Assistant documentaire Altéa")
question = st.text_input("Votre question")

if question:
    with st.spinner("Recherche dans la documentation…"):
        reponse, extraits = repondre(question)
    st.write(reponse)
    st.subheader("Sources")
    for i, e in enumerate(extraits, start=1):
        with st.expander(f"[{i}] {e['source']}, page {e['page']}  (score {e['score']:.2f})"):
            st.write(e["texte"])