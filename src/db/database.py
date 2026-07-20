import sqlite3
import json
import numpy as np
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer

class VocabDatabase:
    def __init__(self, db_path: str = "vocab.db"):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)

        print("[System] Loading SentenceTransformer embedding model...")
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')

        self._create_table()

    def _create_table(self):
        """Initialize DB structure, and store embedding in BLOB form"""
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS vocab (
                word TEXT PRIMARY KEY,
                card_data TEXT,
                embedding BLOB
            )
        ''')
        self.conn.commit()

    def add_word(self, word: str, card_data: dict):
        """Store vocab data and its embedding vecotr into DB"""
        # to calculate similarity, I combine the word itself and it def as a text
        # Simpified it: consider only the first meaning (Sense) of the word
        definity = ""
        senses = card_data.get("senses", [])
        if senses:
            definition = senses[0].get("definition", "")

        text_to_embed = f"Word: {word}. Definition: {definition}"

        embedding = self.embedder.encode(text_to_embed)
        embedding_bytes = embedding.astype(np.float32).tobytes()

        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO vocab (word, card_data, embedding)
            VALUES (?, ?, ?)
        ''', (word, json.dumps(card_data, ensure_ascii=False), embedding_bytes)
        )
        self.conn.commit()

    def get_similar_words(self, target_word: str, top_k: int = 5, threshold: float = 0.3) -> List[Tuple[str, float]]:
        """Use Cosine Similarity to find the similar words"""
        target_word = target_word.lower().strip()
        cursor = self.conn.cursor()

        # get the target_word vector
        cursor.execute('SELECT embedding FROM vocab WHERE word = ?', (target_word,))
        row = cursor.fetchone()
        if not row:
            return []

        target_emb = np.frombuffer(row[0], dtype=np.float32)

        # get all the other words and their vector
        cursor.execute('SELECT word, embedding FROM vocab WHERE word != ?', (target_word,))
        all_rows = cursor.fetchall()

        if not all_rows:
            return []

        # calculate the cosine similarity (done in AI-Engineering-From-Scratch already)
        results = []
        for r_word, r_emb_bytes in all_rows:
            r_emb = np.frombuffer(r_emb_bytes, dtype=np.float32)

            sim = np.dot(target_emb, r_emb) / (np.linalg.norm(target_emb) * np.linalg.norm(r_emb))

            if sim >= threshold:
                results.append((r_word, float(sim)))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def get_word(self, word: str) -> dict:
        """Fetch the JSON data according to the word (if exists)"""
        cursor = self.conn.cursor()
        cursor.execute('SELECT card_data FROM vocab WHERE word = ?', (word,))
        row = cursor.fetchone()
        return json.loads(row[0]) if row else None

# For Test
if __name__ == "__main__":
    db = VocabDatabase("test_vocab.db")

    mock_data = [
        {
            "word": "mitigate",
            "cefr_level": "C1",
            "senses": [{"part_of_speech": "verb", "definition": "Make a situation less severe, harmful, or painful"}],
            "synonyms": ["alleviate", "reduce", "diminish"],
            "antonyms": ["aggravate", "intensify"],
            "example_sentence": "Emergency funds are being provided to help mitigate the effects of the disaster."
        },
        {
            "word": "alleviate",
            "cefr_level": "C1",
            "senses": [{"part_of_speech": "verb", "definition": "Make suffering, deficiency, or a problem less severe"}],
            "synonyms": ["ease", "relieve", "mitigate"],
            "antonyms": ["aggravate", "worsen"],
            "example_sentence": "He put on his sunglasses, which did little to alleviate the sun's glare."
        },
        {
            "word": "exacerbate",
            "cefr_level": "C2",
            "senses": [{"part_of_speech": "verb", "definition": "Make a problem, bad situation, or negative feeling worse"}],
            "synonyms": ["aggravate", "worsen", "inflame"],
            "antonyms": ["mitigate", "alleviate"],
            "example_sentence": "The exorbitant cost of land in urban areas only exacerbated the problem."
        },
        {
            "word": "ephemeral",
            "cefr_level": "C2",
            "senses": [{"part_of_speech": "adjective", "definition": "Lasting for a very short time"}],
            "synonyms": ["transitory", "transient", "fleeting"],
            "antonyms": ["permanent", "eternal"],
            "example_sentence": "Fashions are ephemeral, but true style is timeless."
        }
    ]

    print("[System] Inserting fully compliant mock words to database...")
    for data in mock_data:
        db.add_word(data["word"], data)

    target = "mitigate"
    print(f"\n[System] Finding words similar to: {target}")
    similar_words = db.get_similar_words(target)

    for word, score in similar_words:
        print(f"- {word:12} (Similarity: {score:.4f})")