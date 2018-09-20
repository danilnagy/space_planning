import json, random
from space_planning import get_layout

json_path = "room_data.json"
with open(json_path, 'rb') as f:
    definition = json.loads(f.read())
    room_def = definition["rooms"]

num_rooms = len(room_def)

split_list = [random.random() for i in range(num_rooms-2)]
dir_list = [int(round(random.random())) for i in range(num_rooms-1)]
room_order = range(num_rooms)
random.shuffle(room_order)

min_opening = 3

print "\nINPUTS:"
print "split list:", [round(s, 2) for s in split_list]
print "split direction:", dir_list
print "room order:", room_order

edges_out, adjacency_score, aspect_score = get_layout(room_def, split_list, dir_list, room_order, min_opening)

print "\nOUTPUTS:"
# print edges_out
print "adjacency score:", adjacency_score
print "aspect score:", round(aspect_score, 2)