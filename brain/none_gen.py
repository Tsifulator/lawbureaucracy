"""Search-only engine: retrieval works, no LLM writes an answer.

Used by the cloud deploy, where an 8B model on a shared CPU would be unusably
slow. answer.py already streams the matched decisions BEFORE consulting the
engine, so the user still gets the useful half — this just replaces the
"go run ollama pull" note with something true for a hosted build.

Same interface as ollama_gen.py / gemini.py.
"""

NAME = "search-only"


def available() -> bool:
    return False


def unavailable_msg() -> str:
    return (
        "🔎 **Λειτουργία αναζήτησης.** Αυτή η έκδοση εντοπίζει τις πιο σχετικές "
        "αποφάσεις της ΕΑΔΗΣΥ για την ερώτησή σου — τις βλέπεις παρακάτω με τον "
        "αριθμό τους και σύνδεσμο στο PDF. Οι αυτόματες απαντήσεις AI δεν είναι "
        "ενεργές εδώ."
    )


def rewrite_query(q: str) -> str:
    return q


def contextualize(question: str, history=None) -> str:
    return question


def stream_answer(prompt: str):
    return iter(())
