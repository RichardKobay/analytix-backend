from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="bhadresh-savani/distilbert-base-uncased-emotion"
)

prediction = classifier(
    "I love using transformers. The best part is wide range of support and its easy to use",
)


print(prediction)
