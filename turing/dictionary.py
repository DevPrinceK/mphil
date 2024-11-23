# Dictionary representing a sample student profile
student_profile = {
    "name": "John Doe",
    "age": 21,
    "major": "Computer Science",
    "courses": ["Algorithms", "Data Structures", "Machine Learning"],
    "graduation_year": 2025,
    "gpa": 3.8,
    "contact_info": {
        "email": "johndoe@example.com",
        "phone": "123-456-7890"
    },
    "skills": {
        "programming_languages": ["Python", "Java", "C++"],
        "tools": ["Git", "Docker", "AWS"],
        "soft_skills": ["teamwork", "communication", "problem-solving"]
    }
}


print(dir(student_profile))

# get items: key and value from a dict
print("\nKEY, VALUE PAIR:")
for key, value in student_profile.items():
    print(f"{key}: {value}")

print("\nONLY KEYS:")
# get just the keys
for k in student_profile.keys():
    print(f"{k}")

print("\nONLY VALUES:")
# get just the values
for v in student_profile.values():
    print(f"{v}")

print("\nREMOVE KEY, VALUES (POP):")
popped = student_profile.pop("name", None)
popped2 = student_profile.pop("status", None)
print(f"Item popped: {popped}")
print(f"Item popped: {popped2}")

print("\nREMOVE KEY, VALUES (POPITEM):")
popped = student_profile.popitem()
print(f"Item popped item: {popped}")

print("\nCOPY ITEMS FROM ONE DICT TO ANOTHER (SHALLOW COPY):")
'''values in copied dict are dependent on values in original dict'''
new_student = student_profile.copy()

print("\nCOPY ITEMS FROM ONE DICT TO ANOTHER (DEEP COPY):")
'''values in copied dict are not dependent on values in original dict'''
import copy
new_student = copy.deepcopy(student_profile)