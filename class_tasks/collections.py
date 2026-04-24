from collections import Counter, defaultdict, OrderedDict

#The collections module in Python is a built-in library that provides specialized container data types designed to be more efficient and functional than general-purpose types like list, dict, set, and tuple

# ordered dict
d = OrderedDict()
d['first'] = 1
d['second'] = 2
d['third'] = 3

# Move 'first' to the very end
d.move_to_end('first')

print(list(d.keys()))
# Output: ['second', 'third', 'first']

# Equality check is order-sensitive
d2 = OrderedDict([('second', 2), ('third', 3), ('first', 1)])
print(d == d2) # True


# default dict

# Use 'list' as the default factory
student_grades = [('Alice', 'A'), ('Bob', 'B'), ('Alice', 'C')]
grouped_grades = defaultdict(list)

for name, grade in student_grades:
    # No need to check if the name exists; defaultdict handles it
    grouped_grades[name].append(grade)

print(dict(grouped_grades))
# Output: {'Alice': ['A', 'C'], 'Bob': ['B']}

# counter

words = ['apple', 'orange', 'apple', 'pear', 'orange', 'apple']

# Standard dict would need an 'if key not in' loop
# Counter does it automatically:
word_counts = Counter(words)

print(word_counts)  
# Output: Counter({'apple': 3, 'orange': 2, 'pear': 1})

# Get the top 2 most frequent items
print(word_counts.most_common(2)) 
# Output: [('apple', 3), ('orange', 2)]