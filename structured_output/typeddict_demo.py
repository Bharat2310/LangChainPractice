from typing import TypedDict

class person(TypedDict):
    name: str
    age: int

new_person : person = {"name":"hello", "age":2}
print(new_person)