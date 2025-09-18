'''
Count how many times each exercise appears.
Add a new exercise "pull-up" with 3 reps.
Increase "bench" by 2 reps.
Print out all exercises with their counts.
'''

workout = ["squat", "bench", "squat", "deadlift", "bench", "squat"]

exercise_dict: dict = {}

for exercise in workout:
    exercise_dict[exercise] = exercise_dict.get(exercise,0)+1
    
print("initial counts",exercise_dict)

# 2. Add a new exercise
exercise_dict["pull-up"] = 3

# 3. Increase bench by 2 reps
exercise_dict["bench"] += 2

# 4. Print nicely
for exercise, reps in exercise_dict.items():
    print(f"{exercise}: {reps}")