# Script to find students who did not submit an assignment

import os
import pandas as pd

dir_path = "p4"
student_list_xlsx = "CSC 591 (603) SPRG 2019 Grades.xlsx"

submission_dirs = os.listdir(dir_path)
df = pd.read_excel(student_list_xlsx)

df['name'] = (df['Last name'].str.replace(r"[\"\',]", '') + " " + df['First name'])

student_list = set(df['name'])
submission_exists = set()

for dir_name in submission_dirs:
    if ".DS" in dir_name:
        continue
    name = dir_name.split("__")[0]
    submission_exists.add(name.replace("_", " "))

no_submissions = student_list - submission_exists
print ("Total students: ", len(student_list))
print("Number of submissions: ", len(submission_exists))
print ("Missing submissions: ", len(no_submissions))
print ("Students without a submission: ", sorted(no_submissions))
