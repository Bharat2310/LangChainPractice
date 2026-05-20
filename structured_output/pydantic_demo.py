from pydantic import BaseModel, EmailStr
from typing import Optional

class Student(BaseModel):

    name:str="bharat"
    age : Optional[int] = None
    email : EmailStr


new_student = {"age" : 21, "email": "abd@gmail.c"}

student = Student(**new_student)
print(student)