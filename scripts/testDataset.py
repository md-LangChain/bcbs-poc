from langsmith import Client

questions = [
    "what's the weather in sf",
    "what's the weather in san fran",
    "what's the weather in tangier"
]

answers = [
    "It's 60 degrees and foggy.",
    "It's 60 degrees and foggy.",
    "It's 90 degrees and sunny.",
]

ls_client = Client()
dataset = ls_client.create_dataset("weather agent")
ls_client.create_examples(
    inputs=[{"question": q} for q in questions],
    outputs=[{"answer": a} for a in answers],
    dataset_id=dataset.id,
)