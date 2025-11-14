# proof of concept: LLM connection with classification
import sys
sys.path.append("src")
from quizlet_loader import QuizletLoader

loader = QuizletLoader(
    file_path="/Users/jessica/Desktop/github/cs4100-final-project/src/flashcards/high-quality-testcards.json",
    file_format="json",
    classify_cards=True
)

# test initial LLM connection
print("testing LLM connection...")
try:
    result = loader._classify_card("What is precision concerned with? A. Correct positive predictions out of all predictions B. Correct positive predictions out of true positives C. Correct negative predictions out of false negatives D. How many total samples the model predicted", "A")
    print("LLM classification result:", result)

except Exception as e:
    print("LLM connection fail:", e)


# load and classify flashcards with llm
for i, doc in enumerate(loader.lazy_load(), start=1):

    print(f"\nCard {i}:")

    # classification from llm
    flash_type = doc.metadata.get("flashcard_type")
    print("  flashcard_type:", flash_type)

    if "llm_raw_response" in doc.metadata:
        print("  llm_raw_response:", doc.metadata["llm_raw_response"])

    print("  term:", doc.metadata.get("term"))
    print("  definition:", doc.metadata.get("definition"))

    if i >= 20:
        break
