import random

print("Welcome to the Number Guessing Game!")
print("I'm thinking of a number between 1 and 100.")

number = random.randint(1, 100)

guess = int(input("Make a guess: "))

if guess == number:
