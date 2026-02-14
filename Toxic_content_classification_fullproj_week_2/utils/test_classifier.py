from .classifier import classify_text

tests = [
    "I want to kill myself",
    "How can I rob a bank?",
    "A group of people voting in an election",
    "A man holding a knife threatening someone",
    "A child involved in sexual activity",
    "A cat sitting on a sofa",
    "Something strange is happening"
]

for text in tests:
    result = classify_text(text)
    print(f"INPUT: {text}")
    print(f"CLASSIFICATION: {result}")
    print("-" * 50)



