# File: captsone_peergrading_organizer.py
# Author: Ronil Pancholia
# Date: 5/3/19
# Time: 9:53 PM
import random

from easydict import EasyDict

random.seed = 629

team_ids = [4,6,7,8,9,10,16,18,20,24,29,34,35,37,39,5,11,12,13,25,28,41,42,44,1,3,14,15,17,21,23,26,32,38,43,45,19,22,27,30,31,33,36,40,2]
topic_ids = ["1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","2","2","2","2","2","2","2","2","2","3","3","3","3","3","3","3","3","3","3","3","3","4","4","4","4","4","4","4","5","6"]

team_topic_tup = []
for i in range(len(team_ids)):
    team_topic_tup.append((team_ids[i], topic_ids[i]))

topic_counts = EasyDict()
for topic_id in topic_ids:
    count = topic_counts.get(str(topic_id), 0)
    topic_counts[str(topic_id)] = count+1

topic_teams = EasyDict()
for i in range(len(topic_ids)):
    topic = topic_ids[i]
    teams = topic_teams.get(str(topic), [])
    teams.append(team_ids[i])
    topic_teams[str(topic)] = teams

print (topic_counts)
print (topic_teams)

to_grade = EasyDict()

for topic in set(topic_ids):
    teams = topic_teams[str(topic)]
    for i in range(len(teams)):
        team_id = teams[i]
        curr = to_grade.get(str(team_id), [])
        if i+1 < len(teams):
            curr.append(teams[i+1])
        elif teams[0] != team_id:
            curr.append(teams[0])

        to_grade[str(team_id)] = curr

print (to_grade)

random.shuffle(team_ids)
print(team_ids)

for team_id in to_grade.keys():
    curr = to_grade.get(team_id)
    if team_ids[0] not in curr and str(team_id) != str(team_ids[0]):
        curr.append(team_ids[0])
        del team_ids[0]
    elif team_ids[1] not in curr and str(team_id) != str(team_ids[1]):
        curr.append(team_ids[1])
        del team_ids[1]
    elif team_ids[2] not in curr and str(team_id) != str(team_ids[2]):
        curr.append(team_ids[2])
        del team_ids[2]
    else:
        curr.append(team_ids[3])
        del team_ids[3]

    to_grade[team_id] = curr

print(to_grade)

team_ids = [4,6,7,8,9,10,16,18,20,24,29,34,35,37,39,5,11,12,13,25,28,41,42,44,1,3,14,15,17,21,23,26,32,38,43,45,19,22,27,30,31,33,36,40,2]
random.shuffle(team_ids)

for team_id in to_grade.keys():
    curr = to_grade.get(team_id)
    if team_ids[0] not in curr and str(team_id) != str(team_ids[0]):
        curr.append(team_ids[0])
        del team_ids[0]
    elif team_ids[1] not in curr and str(team_id) != str(team_ids[1]):
        curr.append(team_ids[1])
        del team_ids[1]
    elif team_ids[2] not in curr and str(team_id) != str(team_ids[2]):
        curr.append(team_ids[2])
        del team_ids[2]
    else:
        curr.append(team_ids[3])
        del team_ids[3]

    to_grade[team_id] = curr

for i in range(45):
     print(to_grade[str(i+1)])

counts = {}
for i in range(45):
    for j in to_grade[str(i+1)]:
        count = counts.get(str(j), 0)
        counts[str(j)] = count+1

for i in range(45):
     print(i+1, counts[str(i+1)])