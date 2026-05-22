import random

class Normal_LLM():
    def __init__(self):
        print("passed")

    def predict(self, prompt):
        response_list = [
            "Artificial Intelligence",
            "hello world",
            "world hello"
        ]
        return {"response": random.choice(response_list)}
    
llm = Normal_LLM()

print(llm.predict("ello world"))