from pydantic import BaseModel, ValidationError, field_validator
from typing import List, Optional

# Define a Pydantic model
class User(BaseModel):
    id: int
    name: str
    age: Optional[int] = None
    email: str
    friends: List[str] = []

    # Custom validator to ensure age is a positive integer
    @field_validator('age')
    def age_must_be_positive(self, v):
        if v is not None and v <= 0:
            raise ValueError('\n Age must be a positive integer')
        return v

# Example usage
def main():
    # Valid data
    try:
        user = User(id=1, name='John Doe', age=30, email='john.doe@example.com', friends=['Alice', 'Bob'])
        print(user)
    except ValidationError as e:
        print(e.json())

    # Invalid data: age is negative
    try:
        user = User(id=2, name='Jane Doe', age=-5, email='jane.doe@example.com')
    except ValidationError as e:
        print("\n Validation Error:", e.json())

    # Invalid data: missing required field 'email'
    try:
        user = User(id=3, name='Jim Doe')
    except ValidationError as e:
        print("\n Validation Error:", e.json())

if __name__ == "__main__":
    main()