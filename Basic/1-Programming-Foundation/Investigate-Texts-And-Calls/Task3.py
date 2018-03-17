"""
Intro to Python Lab 1, Task 3

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

if __name__ == '__main__':
    prefix_fixed_line = "("
    suffix_fixed_line = ")"
    prefix_of_mobile_number = ["7", "8", "9"]
    area_code_in_Bangalore = "080"
    code_called_from_Bangalore = []
    sum_of_calls_from_Bangalore = 0
    calls_from_Bangalore_to_Bangalore = 0

    def is_telephone_number_in_Bangalore(telephone_number):
        if telephone_number[0] == prefix_fixed_line:
            code_area = telephone_number[1:4]
        else:
            code_area = telephone_number
        if code_area == area_code_in_Bangalore:
            return True
        return False

    def process_telephone_number(telephone_number):
        """
        if a telephone_number is fixed line, return the number in the parentheses
        if a telephone_number is mobile number, return the prefix that is the first four numbers
        """
        if telephone_number[0] == prefix_fixed_line:
            return telephone_number.split(suffix_fixed_line, 1)[0][1:]
        elif telephone_number[0] in prefix_of_mobile_number:
            return telephone_number[0:4]
        return 0

    for call in calls:
        if is_telephone_number_in_Bangalore(call[0]):
            sum_of_calls_from_Bangalore += 1
            result = process_telephone_number(call[1])
            if result and (result not in code_called_from_Bangalore):
                code_called_from_Bangalore.append(result)
            if is_telephone_number_in_Bangalore(result):
                calls_from_Bangalore_to_Bangalore += 1

    code_called_from_Bangalore = sorted(code_called_from_Bangalore, reverse=False)
    percentage = float(calls_from_Bangalore_to_Bangalore) / float(sum_of_calls_from_Bangalore) * 100
    percentage = "{0:.2f}".format(percentage)
    partA_message = "The numbers called by people in Bangalore have codes:"
    partB_message = "<{0}%> percent of calls from fixed lines in Bangalore are calls to other fixed lines in Bangalore."\
        .format(percentage)
    for code in code_called_from_Bangalore:
        partA_message += "\n<{}>".format(code)

    print partA_message
    print partB_message
"""
TASK 3:
(080) is the area code for fixed line telephones in Bangalore. 
Fixed line numbers include parentheses, so Bangalore numbers 
have the form (080)xxxxxxx.)

Part A: Find all of the area codes and mobile prefixes called by people
in Bangalore. 
 - Fixed lines start with an area code enclosed in brackets. The area 
   codes vary in length but always begin with 0.
 - Mobile numbers have no parentheses, but have a space in the middle
   of the number to help readability. The prefix of a mobile number
   is its first four digits, and they always start with 7, 8 or 9.
 - Telemarketers' numbers have no parentheses or space, but they start
   with the area code 140.

Print the answer as part of a message:
"The numbers called by people in Bangalore have codes:"
 <list of codes>
The list of codes should be print out one per line in lexicographic order with no duplicates.

Part B: What percentage of calls from fixed lines in Bangalore are made
to fixed lines also in Bangalore? In other words, of all the calls made
from a number starting with "(080)", what percentage of these calls
were made to a number also starting with "(080)"?

Print the answer as a part of a message::
"<percentage> percent of calls from fixed lines in Bangalore are calls
to other fixed lines in Bangalore."
The percentage should have 2 decimal digits
"""
