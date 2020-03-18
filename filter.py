#
# 3/6
# Filters out non-useful comments.
#

import json

starting_year = 2013
num_years = 7

validation_data = []
test_data = []
train_data = []
data_points = [[], []]

for y in range(num_years):
    year = starting_year + y
    print(year)

    data = json.loads(open("comments/{}.json".format(year)).read())["data"]
    posts = {}
    counter = 0

    for row in data:
        if (
            "selftext" in row["post"] and row["post"]["selftext"] not in ("[deleted]", "[removed]")
        ) and (
            "body" in row["comment"] and row["comment"]["body"] not in ("[deleted]", "[removed]")
        ) and (
            "body" in row["reply"] and row["reply"]["body"] not in ("[deleted]", "[removed]")
        ):
            label = (
                "Δ" in row["reply"]["body"]
            ) or (
                "!delta" in row["reply"]["body"].lower()
            ) or (
                "#8710;" in row["reply"]["body"]
            )

            point = {
                "id": counter,
                "post_id": row["post"]["id"],
                "title": row["post"]["title"],
                "post": row["post"]["selftext"],
                "comment": row["comment"]["body"],
                "label": label
            }
            data_points[label].append(point)
            counter += 1

        # diffs = []
        # for row in rows:
        #     diff = row["comment"]["score"] - row["reply"]["score"]
        #     diffs.append((diff, row))

        # # Balance dataset
        # diffs.sort(key=lambda x: abs(x[0]), reverse=True)
        # diffs = diffs[:len(diffs) // scale_pts * scale_pts]

        # diffs.sort(key=lambda x: x[0])
        # for i in range(scale_pts):
        #     label = i - scale_pts // 2
        #     partition_size = len(diffs) // scale_pts

        #     for diff in diffs[i * partition_size:(i + 1) * partition_size]:
        #         row = diff[1]
                # point = {
                #     "id": row["post"]["id"] + "_" + row["comment"]["id"],
                #     "title": row["post"]["title"],
                #     "post": row["post"]["selftext"],
                #     "comment": row["comment"]["body"],
                #     "reception": label
                # }
        #         data_points[i].append(point)

print(len(data_points[0]))
delta_0_write = json.dumps({"data": data_points[0]})
with open("dataset/0.json", "w") as out_file:
    out_file.write(delta_0_write)

print(len(data_points[1]))
delta_1_write = json.dumps({"data": data_points[1]})
with open("dataset/1.json", "w") as out_file:
    out_file.write(delta_1_write)

# partition_size = int(0.2 * len(data_points[1]))
# for reception_points in data_points:
#     validation_data += reception_points[:partition_size]
#     test_data += reception_points[partition_size:2 * partition_size]
#     train_data += reception_points[2 * partition_size:]

# print(len(validation_data))
# validation_write = json.dumps({"data": validation_data})
# with open("dataset/validation.json", "w") as out_file:
#     out_file.write(validation_write)

# print(len(test_data))
# test_write = json.dumps({"data": test_data})
# with open("dataset/test.json", "w") as out_file:
#     out_file.write(test_write)

# print(len(train_data))
# train_write = json.dumps({"data": train_data})
# with open("dataset/train.json", "w") as out_file:
#     out_file.write(train_write)
