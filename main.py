import os
import time
from pathlib import Path
from datetime import datetime

import chromadb
from dotenv import load_dotenv
from google import genai
from google.genai import types
from docx import Document

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
MODEL =  "gemini-3.5-flash-lite"
EMBEDDING_MODEL = "gemini-embedding-001"

DOCUMENTS_DIR = Path("documents")
DB_DIR = "chroma_db"
REPORT_FILE = "results_rag.docx"
COLLECTION_NAME = "documents"

TEST_QUESTIONS = [
    "Що таке нейронна мережа?",
    "Які основні компоненти нейронної мережі?",
    "Що таке машинне навчання?",
    "Які основні типи машинного навчання існують?",
    "Що таке генеративний штучний інтелект?"
]

UNKNOWN_QUESTION = "Яка столиця Японії?"

if not API_KEY:
    raise ValueError("Не знайдено GEMINI_API_KEY у файлі .env")

client = genai.Client(
    api_key=API_KEY,
    http_options=types.HttpOptions(timeout=120000)
)


def split_text(text, size=1000, overlap=150):
    text = " ".join(text.split())
    return [
        text[i:i + size]
        for i in range(0, len(text), size - overlap)
        if text[i:i + size].strip()
    ]


def get_embedding(text, task):
    result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
        config=types.EmbedContentConfig(task_type=task)
    )
    return result.embeddings[0].values


def load_documents():
    files = list(DOCUMENTS_DIR.glob("*.txt"))

    if not files:
        raise FileNotFoundError(
            "У папці documents не знайдено текстових файлів .txt"
        )

    chunks = []
    metadatas = []
    ids = []

    for file in files:
        text = file.read_text(encoding="utf-8")
        for number, chunk in enumerate(split_text(text)):
            chunks.append(chunk)
            metadatas.append({"source": file.name})
            ids.append(f"{file.stem}_{number}")

    return chunks, metadatas, ids


def create_database():
    chunks, metadatas, ids = load_documents()
    db = chromadb.PersistentClient(path=DB_DIR)
    collection = db.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    embeddings = []
    for chunk in chunks:
        embeddings.append(get_embedding(chunk, "RETRIEVAL_DOCUMENT"))

    collection.upsert(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas
    )

    print(f"Завантажено фрагментів документів: {len(chunks)}")
    return collection


def search_documents(collection, question, count=4):
    question_embedding = get_embedding(question, "RETRIEVAL_QUERY")

    result = collection.query(
        query_embeddings=[question_embedding],
        n_results=min(count, collection.count())
    )

    fragments = []
    for text, metadata, distance in zip(
        result["documents"][0],
        result["metadatas"][0],
        result["distances"][0]
    ):
        similarity = max(0, 1 - distance)
        fragments.append({
            "text": text,
            "source": metadata["source"],
            "similarity": similarity
        })

    return fragments


def ask_gemini(prompt):
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    max_output_tokens=1000
                )
            )

            if response.text:
                return response.text.strip()

            print("Gemini повернув порожню відповідь.")

        except Exception as error:
            print(f"Помилка Gemini: {error}")

            error_text = str(error)

            if any(code in error_text for code in ["429", "500", "502", "503", "504"]):
                if attempt < 2:
                    print("Повторна спроба через 10 секунд...")
                    time.sleep(10)
                    continue

            return "Не вдалося отримати відповідь від Gemini."

    return "Не вдалося отримати відповідь від Gemini після трьох спроб."

def show_fragments(fragments):
    print("\nЗНАЙДЕНІ ФРАГМЕНТИ")
    for i, item in enumerate(fragments, 1):
        print(f"\n{i}. Файл: {item['source']}")
        print(f"Схожість: {item['similarity']:.2%}")
        print(item["text"])


def rag_answer(collection, question):
    fragments = search_documents(collection, question)
    show_fragments(fragments)

    if not fragments or fragments[0]["similarity"] < 0.35:
        answer = "У наданих документах недостатньо інформації для відповіді."
        print("\nВідповідь:", answer)
        return answer, fragments

    context = "\n\n".join(item["text"] for item in fragments)

    prompt = f"""
Ти помічник, який відповідає на запитання лише на основі наданого контексту.

Контекст:
{context}

Запитання: {question}

Якщо контекст не містить потрібної інформації, прямо скажи:
"У наданих документах недостатньо інформації для відповіді."
Не вигадуй факти. Відповідай українською мовою.
"""

    answer = ask_gemini(prompt)
    print("\nВідповідь RAG:", answer)
    return answer, fragments


def compare_answers(collection, question):
    print("\nВІДПОВІДЬ ЗВИЧАЙНОЇ LLM")
    prompt = (
        "Дай відповідь українською мовою на запитання. "
        "Відповідай зрозуміло та по суті:\n" + question
    )
    llm_answer = ask_gemini(prompt)
    print(llm_answer)

    print("\nВІДПОВІДЬ RAG")
    rag_result, fragments = rag_answer(collection, question)

    report = (
        f"Запитання: {question}\n\n"
        f"LLM без RAG:\n{llm_answer}\n\n"
        f"RAG:\n{rag_result}\n\n"
        f"{format_fragments(fragments)}"
    )
    save_result("Порівняння LLM та RAG", report)


def format_fragments(fragments):
    if not fragments:
        return "Знайдених фрагментів немає."

    lines = ["Знайдені фрагменти:"]
    for item in fragments:
        lines.append(
            f"{item['source']} | схожість: {item['similarity']:.2%}\n"
            f"{item['text']}"
        )
    return "\n\n".join(lines)


def save_result(title, content):
    if Path(REPORT_FILE).exists():
        doc = Document(REPORT_FILE)
    else:
        doc = Document()
        doc.add_heading("Звіт з лабораторної роботи RAG", 0)

    doc.add_heading(title, level=1)
    doc.add_paragraph(datetime.now().strftime("%d.%m.%Y %H:%M"))
    doc.add_paragraph(content)

    try:
        doc.save(REPORT_FILE)
        print(f"\nРезультат додано до {REPORT_FILE}")
    except PermissionError:
        print(f"\nНе вдалося зберегти звіт. Закрий файл {REPORT_FILE} у Word і повтори дію.")


def test_questions(collection):
    results = []

    for number, question in enumerate(TEST_QUESTIONS, 1):
        print(f"\nТЕСТ {number}/{len(TEST_QUESTIONS)}")
        print("Запитання:", question)
        answer, fragments = rag_answer(collection, question)
        results.append(
            f"Тест {number}\n"
            f"Запитання: {question}\n"
            f"Відповідь: {answer}\n"
            f"{format_fragments(fragments)}"
        )

    save_result("Тестування на п'яти запитаннях", "\n\n".join(results))


def test_unknown_question(collection):
    print("\nПеревірка запитання, якого немає в документах:")
    answer, fragments = rag_answer(collection, UNKNOWN_QUESTION)

    save_result(
        "Перевірка відсутньої інформації",
        f"Запитання: {UNKNOWN_QUESTION}\n"
        f"Відповідь: {answer}\n"
        f"{format_fragments(fragments)}"
    )


def main():
    print("ЛАБОРАТОРНА РОБОТА")
    print("Retrieval-Augmented Generation (RAG)")

    collection = create_database()

    while True:
        print("\nМЕНЮ")
        print("1. Задати запитання до RAG")
        print("2. Порівняти LLM без RAG та з RAG")
        print("3. Провести тестування на п'яти запитаннях")
        print("4. Перевірити запитання без відповіді в документах")
        print("5. Вийти")

        choice = input("\nОберіть дію: ").strip()

        if choice == "1":
            question = input("\nВведіть запитання: ").strip()
            if question:
                answer, fragments = rag_answer(collection, question)
                save_result(
                    "Запитання до RAG",
                    f"Запитання: {question}\n"
                    f"Відповідь: {answer}\n"
                    f"{format_fragments(fragments)}"
                )
        elif choice == "2":
            question = input("\nВведіть запитання для порівняння: ").strip()
            if question:
                compare_answers(collection, question)
        elif choice == "3":
            test_questions(collection)
        elif choice == "4":
            test_unknown_question(collection)
        elif choice == "5":
            print("Роботу завершено.")
            break
        else:
            print("Неправильний пункт меню. Спробуйте ще раз.")


if __name__ == "__main__":
    main()
