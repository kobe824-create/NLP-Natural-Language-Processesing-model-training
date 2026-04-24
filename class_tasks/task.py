# # numbers = [1, 12, 6, 8]
# # sum = 0

# # for i in numbers:
# #     if i % 2 == 0:
# #        sum += i
# # print(sum)       

# # # funtion for reversing a given string without built-in reverse 

# # string = "happy"
# # reversed_string = ""

# # for i in range(len(string) - 1, - 1, - 1):
# #     reversed_string += string[i]

# # print(reversed_string)


# # # a python program that counts the number of words in a sentence.

# # sentence = "I am a passionate developer"
# # print(len(sentence))


# # It was cool.

# # word = "again"
# # r_word = ""

# # for i in word:
# #     r_word += i + word

# # print(r_word)

# num_1 = int(input("First number: "))
# num_2 = int(input("Second number: "))
# operator = input("Enter operation (+, -, *, /): ")

# if operator == "+":
#     print(num_1 + num_2)

# elif operator == "-":
#     print(num_1 - num_2)

# elif operator == "*":
#     print(num_1 * num_2)

# elif operator == "/":
#     if num_2 == 0:
#         print("Division by zero is not allowed.")
#     else:
#         print(num_1 / num_2)

# else:
#     print("Invalid operator")

# # displayinfg the inputed name
# # name = input("my name:")
# # print(name)

def cuter_function(x, y):
    def inner_function(x, y):
     return x +  y
    return inner_function(x, y)
result = cuter_function(5, 10)
print(result)
