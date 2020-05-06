# autograding-using-natural-language-understanding
Auto grade the cumulative text responses of the students based on originality, sentiment analysis, effort and constructive in the responses. 

Steps:
1. Extract the data from critviz website using selenium and BeautifulSoup
2. Store the data in excel sheet with two column. One for student name, and second for the response.
3. Calculate the grade based on originality, constructiveness, balanced sentiment analysis, and efforts.

To calculate the grades, change the file directory to folder and:
run autograde.py
Give the url of assignment when prompted
