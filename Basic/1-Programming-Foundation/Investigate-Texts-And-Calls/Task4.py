"""
Intro to Python Lab 1, Task 4

Complete each task in the file for that task. Submit the whole folder
as a zip file or GitHub repo. 
Full submission instructions are available on the Lab Preparation page.
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

    callers = []
    not_telemarketer = set()
    possible_telemarketer = set()

    # numbers of receiving calls are not telemarketers
    for call in calls:
        caller = call[0]
        callers.append(caller)
        not_telemarketer.add(call[1])

    # numbers of sending text or receiving texts are not telemarketers
    for text in texts:
        not_telemarketer.add(text[0])
        not_telemarketer.add(text[1])

    for number in callers:
        if number not in not_telemarketer:
            possible_telemarketer.add(number)

    possible_telemarketer = sorted(possible_telemarketer, reverse=False)
    message = "These numbers could be telemarketers: "
    for number in possible_telemarketer:
        message += "\n{0}".format(number)
    print message

"""
TASK 4:
The telephone company want to identify numbers that might be doing
telephone marketing. Create a set of possible telemarketers: 
these are numbers that make outgoing calls but never send texts,
receive texts or receive incoming calls.

Print a message: 
"These numbers could be telemarketers: "
<list of numbers>
The list of numbers should be print out one per line in lexicographic order with no duplicates.
"""

