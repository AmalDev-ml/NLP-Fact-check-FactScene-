"""
build_faiss_index.py — Build local FAISS index from LIAR and FEVER datasets + Core Verified Facts.
------------------------------------------------------------------------------------------------
Constructs a unified, dense semantic index (all-MiniLM-L6-v2, 384 dims) 
cached to disk at data/faiss_index.bin and data/faiss_metadata.pkl.
"""

import os
import csv
import pickle
import numpy as np
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
FAISS_INDEX_PATH = os.path.join(DATA_DIR, "faiss_index.bin")
FAISS_META_PATH = os.path.join(DATA_DIR, "faiss_metadata.pkl")
LIAR_TRAIN_PATH = os.path.join(DATA_DIR, "liar_train.tsv")

CORE_FACTS = [
    {
        "evidence": "India achieved independence from British rule on 15 August 1947, through the Indian Independence Act 1947 passed by the Parliament of the United Kingdom.",
        "claim": "Independence of India 1947",
        "text": "Independence of India 1947 — India achieved independence from British colonial rule on 15 August 1947, through the Indian Independence Act 1947.",
        "source": "Historical Fact Archive / Wikipedia",
        "url": "https://en.wikipedia.org/wiki/Indian_Independence_Act_1947",
        "label": "true",
        "dataset": "CoreFacts"
    },
    {
        "evidence": "The Earth is an oblate spheroid that revolves around the Sun in an elliptical orbit with a period of approximately 365.25 days.",
        "claim": "The Earth is round and revolves around the Sun.",
        "text": "The Earth is round and revolves around the Sun. — The Earth is an oblate spheroid that orbits the Sun.",
        "source": "Planetary Science / NASA",
        "url": "https://en.wikipedia.org/wiki/Earth",
        "label": "true",
        "dataset": "CoreFacts"
    },
    {
        "evidence": "Scientific consensus firmly rejects the flat Earth model; satellite imagery, astronomical observation, and physical geodesy confirm Earth is spherical.",
        "claim": "The Earth is flat and not a sphere.",
        "text": "The Earth is flat and not a sphere. — Scientific consensus confirms Earth is an oblate spheroid and completely rejects flat Earth hypotheses.",
        "source": "Planetary Science / NASA",
        "url": "https://en.wikipedia.org/wiki/Spherical_Earth",
        "label": "false",
        "dataset": "CoreFacts"
    },
    {
        "evidence": "The Eiffel Tower is a wrought-iron lattice tower located on the Champ de Mars in Paris, France, constructed from 1887 to 1889.",
        "claim": "The Eiffel Tower is located in Paris, France.",
        "text": "The Eiffel Tower is located in Paris, France. — The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France.",
        "source": "Architecture / History",
        "url": "https://en.wikipedia.org/wiki/Eiffel_Tower",
        "label": "true",
        "dataset": "CoreFacts"
    },
    {
        "evidence": "There is no scientific basis linking 5G cellular mobile technology to the spread or biological generation of SARS-CoV-2 or COVID-19.",
        "claim": "5G cellular networks transmit the coronavirus virus.",
        "text": "5G cellular networks transmit the coronavirus virus. — Health agencies and the WHO confirm that radio waves from 5G cannot create or transmit biological viruses.",
        "source": "World Health Organization",
        "url": "https://www.who.int/emergencies/diseases/novel-coronavirus-2019/advice-for-public/myth-busters",
        "label": "false",
        "dataset": "CoreFacts"
    },
    {
        "evidence": "Apollo 11 was the American spaceflight that first landed humans on the Moon on July 20, 1969, with Neil Armstrong and Buzz Aldrin.",
        "claim": "Humans first landed on the Moon in 1969.",
        "text": "Humans first landed on the Moon in 1969. — Apollo 11 landed Neil Armstrong and Buzz Aldrin on the Moon on July 20, 1969.",
        "source": "NASA History",
        "url": "https://en.wikipedia.org/wiki/Apollo_11",
        "label": "true",
        "dataset": "CoreFacts"
    }
]

def build_index(max_fever_samples: int = 400, max_liar_samples: int = 400):
    print("=== Building Local FAISS Evidence Index ===")
    corpus = list(CORE_FACTS)

    # 1. Ingest FEVER samples
    print("Loading FEVER dataset...")
    try:
        fever_df = pd.read_parquet("hf://datasets/pietrolesci/nli_fever/data/dev-00000-of-00001.parquet")
        verifiable = fever_df[fever_df["fever_gold_label"].isin(["SUPPORTS", "REFUTES"])].copy()
        sampled_fever = verifiable.head(max_fever_samples)
        for _, row in sampled_fever.iterrows():
            premise = str(row.get("premise", "")).strip()
            hypothesis = str(row.get("hypothesis", "")).strip()
            label = str(row.get("fever_gold_label", "")).strip()
            if len(premise) > 15 and len(hypothesis) > 5:
                corpus.append({
                    "evidence": premise,
                    "claim": hypothesis,
                    "text": f"{hypothesis} — {premise}",
                    "source": "FEVER Fact Verification Archive",
                    "url": "https://fever.ai/",
                    "label": label,
                    "dataset": "FEVER"
                })
        print(f"Loaded {len(corpus) - len(CORE_FACTS)} records from FEVER.")
    except Exception as e:
        print(f"Warning: Failed to load FEVER dataset: {e}")

    # 2. Ingest LIAR samples
    print("Loading LIAR dataset...")
    if os.path.exists(LIAR_TRAIN_PATH):
        try:
            with open(LIAR_TRAIN_PATH, "r", encoding="utf-8") as f:
                reader = csv.reader(f, delimiter="\t")
                count = 0
                for row in reader:
                    if len(row) >= 16:
                        label = row[2].strip()
                        statement = row[3].strip()
                        subject = row[4].strip()
                        speaker = row[5].strip()
                        justification = row[15].strip() if len(row) > 15 else ""

                        evidence = justification if len(justification) > 20 else f"Statement by {speaker} on {subject}: '{statement}' was evaluated as {label}."
                        if len(statement) > 10:
                            corpus.append({
                                "evidence": evidence,
                                "claim": statement,
                                "text": f"{statement} — {evidence}",
                                "source": f"PolitiFact / LIAR Dataset ({speaker})",
                                "url": "https://www.politifact.com/",
                                "label": label,
                                "dataset": "LIAR"
                            })
                            count += 1
                            if count >= max_liar_samples:
                                break
            print(f"Loaded {count} records from LIAR.")
        except Exception as e:
            print(f"Warning: Failed to load LIAR dataset: {e}")

    print(f"Total corpus size: {len(corpus)} documents.")

    # 3. Embed texts using all-MiniLM-L6-v2
    print("Loading embedding model all-MiniLM-L6-v2...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    texts = [item["text"] for item in corpus]
    print(f"Embedding {len(texts)} texts...")
    embeddings = model.encode(texts, batch_size=128, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)

    # 4. Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings.astype(np.float32))

    # 5. Save index and metadata
    faiss.write_index(index, FAISS_INDEX_PATH)
    with open(FAISS_META_PATH, "wb") as f:
        pickle.dump(corpus, f)

    print(f"Successfully saved FAISS index to {FAISS_INDEX_PATH} ({index.ntotal} vectors).")
    print(f"Successfully saved metadata to {FAISS_META_PATH}.")

if __name__ == "__main__":
    build_index()
