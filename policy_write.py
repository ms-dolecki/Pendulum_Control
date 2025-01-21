import json

array_list = [9.57,18.79,151.3,-68.6]
array_list2 = [[1,2],[3,4]]
policy = {
    "a":array_list
}
file_path = "policy_config.txt"
with open(file_path, 'w') as file:
    print("dumping")
    json.dump(policy, file)

with open(file_path, 'r') as file:
    data1 = json.load(file)
    #data2 = json.load(file)

print(data1["a"])
#print(data2)
print("done")
