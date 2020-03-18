#
# 1/6
# Gathers list of submissions.
#

import datetime as dt
import calendar
import urllib.request as request
import json
from time import sleep

starting_year = 2013
num_years = 7
fields = ("id", "created_utc", "author", "title", "selftext", "score", "num_comments", "url")

for y in range(num_years):
    year = starting_year + y
    submissions_data = []

    for m in range(12):
        month = m + 1
        print(year, month)

        t0 = int(dt.datetime(
                year=year,
                month=month, 
                day=1, 
                hour=0, 
                minute=0, 
                second=0, 
                tzinfo=dt.timezone.utc
            ).timestamp())
        t1 = int(dt.datetime(
                year=year,
                month=month,
                day=calendar.monthrange(year, month)[1],
                hour=23,
                minute=59,
                second=59,
                tzinfo=dt.timezone.utc
            ).timestamp())

        # Get top 500 highest rated posts this month
        url = "https://api.pushshift.io/reddit/submission/search/?after={}&before={}&sort_type=score&sort=desc&subreddit=ChangeMyView&limit=500".format(t0, t1)
        t = 5
        while t > 0:
            try:
                with request.urlopen(url) as req:
                    data = json.loads(req.read().decode())
                t = -1
            except:
                print("Retrying...")
                sleep(t)
                t *= 2

        for submission in data["data"]:
            # Filter by submissions with at least 10 comments
            if submission["num_comments"] >= 10:
                row = {}
                for field in fields:
                    if field in submission:
                        row[field] = submission[field]
                submissions_data.append(row)

    submissions_write = json.dumps({"data": submissions_data})
    with open("submissions/{}.json".format(year), "w") as out_file:
        out_file.write(submissions_write)
