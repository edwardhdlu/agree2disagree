#
# 2/6
# Builds tuples of (post, comment, reply).
#

import urllib.request as request
import json
from time import sleep

starting_year = 2013
num_years = 7
com_fields = ("id", "created_utc", "body", "score")
sub_fields = ("id", "created_utc", "author", "title", "selftext", "score", "num_comments", "url")

for y in range(num_years):
    year = starting_year + y

    submissions_data = json.loads(open("submissions/{}.json".format(year)).read())["data"]
    submissions_data.sort(key=lambda x: x["score"], reverse=True)
    output_data = []

    for idx, submission in enumerate(submissions_data):
        if idx % 50 == 0:
            print("{} {}/{}".format(year, idx, len(submissions_data)))

        # Get 1000 earliest comments in thread
        url = "https://api.pushshift.io/reddit/comment/search/?sort_type=created_utc&sort=asc&link_id={}&limit=1000&q=*".format(submission["id"])
        t = 5
        while t > 0:
            try:
                with request.urlopen(url) as req:
                    comments_data = json.loads(req.read().decode())
                t = -1
            except:
                print("Retrying...")
                sleep(t)
                t *= 2

        top_level_comments = {}
        replies = {}

        for comment in comments_data["data"]:
            # Top level comment
            if (
                "parent_id" in comment and "link_id" in comment and comment["parent_id"] == comment["link_id"]
            ):
                top_level_comments[comment["id"]] = comment

            # OP reply
            if (
                "author" in comment and comment["author"] == submission["author"] and comment["author"] != "[deleted]"
            ):
                replies[comment["parent_id"].split("_")[-1]] = comment

        for comment_id in top_level_comments:
            comment = top_level_comments[comment_id]
            row = {
                "post": {},
                "comment": {},
                "reply": {}
            }
            
            for field in sub_fields:
                if field in submission:
                    row["post"][field] = submission[field]
            for field in com_fields:
                if field in comment:
                    row["comment"][field] = comment[field]

            if comment_id in replies:
                reply = replies[comment_id]
                for field in com_fields:
                    if field in reply:
                        row["reply"][field] = reply[field]

            output_data.append(row)

    comments_write = json.dumps({"data": output_data})
    with open("comments/{}.json".format(year), "w") as out_file:
        out_file.write(comments_write)
