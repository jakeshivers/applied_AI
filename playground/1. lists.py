visits = ["home", "about", "products", "contact", "products", "cart", "checkout"]
'''
Keep the visits in the exact order they occurred.
Be able to access the 3rd page visited.
Add "confirmation" to the end of the list.
Remove the first "products" entry.
'''

list_var: list[int]

visit_number: int = 2

third_visit = visits[visit_number]

print(third_visit)

visits.append("confirmation")

print(visits)

visits.remove("products")

print(visits)


