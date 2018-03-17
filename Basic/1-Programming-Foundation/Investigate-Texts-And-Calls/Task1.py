"""
Intro to Python Project 1, Task 1

Complete each task in the file for that task. Submit the whole folder
as a zip file or GitHub repo. 
Full submission instructions are available on the Project Preparation page.
"""


"""
Read file into texts and calls. 
It's ok if you don't understand how to read files
You will learn more about reading files in future lesson
"""
import csv
with open('texts.csv', 'r') as f:
    reader = csv.reader(f)
    texts = list(reader)

with open('calls.csv', 'r') as f:
    reader = csv.reader(f)
    calls = list(reader)

    def process_telephone_number(sending_telephone_number, receiving_telephone_number, telephones, number):
        if sending_telephone_number not in telephones:
            telephones.append(sending_telephone_number)
            number += 1
        if receiving_telephone_number not in telephones:
            telephones.append(receiving_telephone_number)
            number += 1
        return telephones, number

    all_telephones = []
    numbers = 0
    for text in texts:
        all_telephones, numbers = process_telephone_number(text[0], text[1], all_telephones, numbers)
    for call in calls:
        all_telephones, numbers = process_telephone_number(call[0], call[1], all_telephones, numbers)
    print "There are {} different telephone numbers in the records.".format(numbers)

"""
TASK 1: 
How many different telephone numbers are there in the records? 
Print a message: 
"There are <count> different telephone numbers in the records."
"""
