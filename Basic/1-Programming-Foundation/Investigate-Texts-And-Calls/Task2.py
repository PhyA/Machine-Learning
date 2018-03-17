"""
Intro to Python Lab 1, Task 2

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


    def get_calling_time(telephone_number, all_time_dictionary, calling_time, longest_time, telephone_number_of_longest_time):
        if all_time_dictionary.get(telephone_number) is None:
            all_time_dictionary[telephone_number] = calling_time
        else:
            all_time_dictionary[telephone_number] += calling_time
            if all_time_dictionary[telephone_number] > longest_time:
                longest_time = all_time_dictionary[telephone_number]
                telephone_number_of_longest_time = telephone_number
        return all_time_dictionary, longest_time, telephone_number_of_longest_time

    total_time = 0
    telephone_time = {}
    longest_time = 0
    longest_telephone_number = 0
    for call in calls:
        call_time = int(call[3])
        telephone_time, total_time, longest_telephone_number = \
            get_calling_time(call[0], telephone_time, call_time, total_time, longest_telephone_number)

        telephone_time, total_time, longest_telephone_number = \
            get_calling_time(call[1], telephone_time, call_time, total_time, longest_telephone_number)

    print "<{}> spent the longest time, <{}> seconds, on the phone during September 2016.".\
        format(longest_telephone_number, total_time)

"""
TASK 2: Which telephone number spent the longest time on the phone
during the period? Don't forget that time spent answering a call is
also time spent on the phone.
Print a message: 
"<telephone number> spent the longest time, <total time> seconds, on the phone during 
September 2016.". 

HINT: Build a dictionary with telephone numbers as keys, and their
total time spent on the phone as the values. You might find it useful
to write a function that takes a key and a value and modifies a 
dictionary. If the key is already in the dictionary, add the value to
the key's existing value. If the key does not already appear in the
dictionary, add it and set its value to be the given value.
"""

