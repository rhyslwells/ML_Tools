from pydantic import BaseModel, ValidationError, Field, field_validator
from typing import List, Optional
from datetime import datetime
from uuid import UUID

# 1. Nested Models
class Friend(BaseModel):
    name: str
    email: str

# 2. Main User Model
class User(BaseModel):
    # Basic fields with type annotations
    id: UUID  # Custom UUID type
    name: str
    age: Optional[int] = None  # Optional field with a default value
    email: str
    friends: List[Friend] = []  # Nested model list with default value
    is_active: bool = True  # Field with a default value
    created_at: datetime = datetime.now  # Dynamic default value using callable
    email_address: str = Field(..., alias="email")  # Field alias for parsing

    # 3. Custom Validators
    @field_validator('age')
    def age_must_be_positive(cls, v):
        """Ensure age is a positive integer."""
        if v is not None and v <= 0:
            raise ValueError('\n Age must be a positive integer')
        return v

    @field_validator('email')
    def validate_email(cls, v):
        """Ensure email belongs to the 'example.com' domain."""
        if '@example.com' not in v:
            raise ValueError('\n Email must belong to example.com domain')
        return v

    # 4. Custom Method
    def greet(self):
        """Return a greeting message."""
        return f"Hello, my name is {self.name}."

    # 5. Configuration Options
    class Config:
        title = "User Model"
        anystr_strip_whitespace = True  # Automatically strip whitespace from strings
        strict = True  # Enforce strict typing for fields

# 6. Inherited Model
class AdminUser(User):
    permissions: List[str] = []  # Additional field for admin-specific permissions

# Main script demonstrating usage
def main():
    # Valid data example
    try:
        user = User(
            id="123e4567-e89b-12d3-a456-426614174000",
            name=" John Doe ",  # Leading and trailing spaces will be stripped
            age=30,
            email="john.doe@example.com",
            friends=[
                {"name": "Alice", "email": "alice@example.com"},
                {"name": "Bob", "email": "bob@example.com"}
            ]
        )
        print("\nValid User Object:")
        print(user.model_dump_json(indent=4))  # Pretty-print JSON representation
        print("\nGreeting:", user.greet())
    except ValidationError as e:
        print("\nValidation Error (Valid Data):")
        print(e.json())

    # Invalid data example: age is negative
    try:
        user = User(
            id="123e4567-e89b-12d3-a456-426614174000",
            name="Jane Doe",
            age=-5,
            email="jane.doe@example.com"
        )
    except ValidationError as e:
        print("\nValidation Error (Negative Age):")
        print(e.json())

    # Invalid data example: incorrect email domain
    try:
        user = User(
            id="123e4567-e89b-12d3-a456-426614174000",
            name="Jim Doe",
            age=25,
            email="jim.doe@notexample.com"
        )
    except ValidationError as e:
        print("\nValidation Error (Invalid Email):")
        print(e.json())

    # Parsing raw data example
    try:
        raw_data = '{"id": "123e4567-e89b-12d3-a456-426614174000", "name": "Arthur Curry", "email": "arthur@example.com"}'
        user = User.model_validate_json(raw_data)
        print("\nParsed User Object:")
        print(user)
    except ValidationError as e:
        print("\nValidation Error (Parsing Raw Data):")
        print(e.json())

    # Demonstrate inherited model (AdminUser)
    try:
        admin = AdminUser(
            id="123e4567-e89b-12d3-a456-426614174000",
            name="Admin User",
            email="admin@example.com",
            permissions=["read", "write", "delete"]
        )
        print("\nAdmin User Object:")
        print(admin.model_dump_json(indent=4))
    except ValidationError as e:
        print("\nValidation Error (Admin User):")
        print(e.json())

if __name__ == "__main__":
    main()
