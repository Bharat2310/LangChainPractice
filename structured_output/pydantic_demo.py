from pydantic import BaseModel, EmailStr, Field
from typing import Optional

class Student(BaseModel):

    name:str="bharat"
    age : Optional[int] = None
    email : EmailStr
    cgpa : float = Field(gt = 0, lt = 10, default=7, description="marks out of 10")


new_student = {"age" : 21, "email": "abd@gmail.c", "cgpa":9}

student = Student(**new_student)
# print(dict(student))
student_json = student.model_dump_json()

print(student_json)