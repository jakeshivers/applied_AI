'''
👉 Your task:

Remove duplicates so each exercise only appears once.
Quickly check if "overhead press" is in the workout.
Add "overhead press" to the workout.

'''

exercises = ["squat", "bench", "deadlift", "squat", "bench", "pull-up"]
exercises_set = set(exercises)
print(exercises_set)

exercise: str = "wall squats"


if exercise not in exercises_set:
    exercises_set.add(exercise)
else:
    print("has this exercise?", exercise in exercises_set)

print(exercises_set)