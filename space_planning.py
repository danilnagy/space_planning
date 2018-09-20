import math, json, random


def remap(value, min1, max1, min2, max2):
    return float(min2) + (float(value) - float(min1)) * (float(max2) - float(min2)) / (float(max1) - float(min1))

def get_split(lengths, split_point):
    if split_point > sum(lengths):
        print "Error: split point beyond length of edge"
        return None
    
    total = 0.0
    
    for i,l in enumerate(lengths):
        if split_point < total + l:
            return i, (split_point - total)/l
        total += l


class Node:
    
    def __init__(self, id):
        self.id = id
        self.children = []
        
    def subdivide(self, dir, id):
        self.children = [Node(self.id), Node(id)]
        self.dir = dir
        return self.children
    
    def collect_areas(self, rooms_list):
        if len(self.children) == 0:
            self.area = rooms_list[self.id]["area"]
            return self.area
        else:
            self.area = sum([child.collect_areas(rooms_list) for child in self.children])
            return self.area
    
    def get_info(self):
        return [self.id, self.dir, self.get_split_ratio()]
    
    def get_split_ratio(self):
        return float(self.children[0].area) / sum([child.area for child in self.children])
    
    def __str__(self):
        if len(self.children) > 0:
            return "[{}, {}]".format(*[str(child) for child in self.children])
        else:
            return str(self.id)

class Edge:
    
    def __init__(self, _id, p1, p2, _f):
        
        self.id = _id
        self.points = [p1,p2]
        self.length = self.get_length()
        self.faces = _f
    
    def get_length(self):
        p1 = self.points[0]
        p2 = self.points[1]
        return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2) ** .5
    
    def copy(self, _id):
        return Edge(_id, self.points[0], self.points[1], [face for face in self.faces])
    
    def get_pt(self, factor):
        p1 = self.points[0]
        p2 = self.points[1]
        
        x = p1[0] + (p2[0]-p1[0]) * factor
        y = p1[1] + (p2[1]-p1[1]) * factor
        
        return [x, y]
    
    def replace_pt(self, pos, _pt):
        self.points[pos] = _pt
        self.length = self.get_length()
    
    def swap_face(self, faceA, faceB):
        self.faces = [faceB if x==faceA else x for x in self.faces]
    
    def get_geo(self, connecting_edge, min_opening):
        
        if connecting_edge:
            l = self.get_length()
            
            p3 = self.get_pt(((l - min_opening)/2.0)/l)
            p4 = self.get_pt(((l + min_opening)/2.0)/l)
            
            return [[self.points[0], p3], [p4,self.points[1]]]
        else:
            return [self.points]
    
    def __str__(self):
        return "Edge[{}] = faces: {}, coords: {}, length: {}".format(self.id, [face.id for face in self.faces], self.points, self.length)
        
class Face:
    
    def __init__(self, _id):
        self.id = _id
        self.name = ""

    def set_edges(self, _e):
        self.edges = _e
    
    def replace_edge(self, dir, pos, edge):
        self.edges[dir][pos] = edge
    
    def add_edge(self, dir, ref_edge, _e):
        for edge in self.edges[dir]:
            if ref_edge in edge:
                edge.insert(edge.index(ref_edge)+1,_e)
    
    def get_edges(self):
        edges = []
        for set in self.edges:
            for comp in set:
                edges += comp
        return [edge.id for edge in edges]
    
    def get_aspect(self):
#        edges = []
        xs = []
        ys = []
        
        for set in self.edges:

            for comp in set:
                for edge in comp:
                    pts = edge.get_geo(False, None)
                    p1 = pts[0][0]
                    p2 = pts[0][1]
                    xs += [p1[0], p2[0]]
                    ys += [p1[1], p2[1]]
        
        domain = [max(xs) - min(xs), max(ys) - min(ys)]
        
        return min(domain) / max(domain)
    
    def get_dims(self):
        x = sum([edge.length for edge in self.edges[0][0]])
        y = sum([edge.length for edge in self.edges[1][0]])
        return x, y
    
    def __str__(self):
        return "Face[{}] = {}".format(self.name, [[[comp.id for comp in edge] for edge in set] for set in self.edges])

class Layout:
    
    def __init__(self, dims):
        
        self.faces = []
        self.edges = []
        
        for i in range(2):
            self.faces.append(Face(len(self.faces)))
        
        for x in [0.0,dims[0]]:
            self.edges.append(Edge(len(self.edges), [x, 0.0], [x, dims[1]], [face for face in self.faces]))
        for y in [0.0,dims[1]]:
            self.edges.append(Edge(len(self.edges), [0.0, y], [dims[0], y], [face for face in self.faces]))
        
        e = self.edges
        
        for face in self.faces:
            face.set_edges([[[e[0]],[e[1]]],[[e[2]],[e[3]]]])
    
    def subdivide(self, target_face, dir, split_factor):
        
        # create direction reference list
        dir = [dir, abs(dir-1)]
        
        # get target face for splitting and create new face
#        target_face = self.faces[_id]
        new_face = Face(len(self.faces))
        
        # create lists to store new point and edge data
        new_pts = []
        new_edges = []
        
        # iterate over edges in splitting direction
        for i, edges in enumerate(target_face.edges[dir[0]]):
            
            # get edge segment to split and split value
            edge_lengths = [edge.length for edge in edges]

            split_length = sum(edge_lengths) * split_factor

            n, val = get_split(edge_lengths, split_length)
            
            # get edge segment to split and all following edge segments
            edge_to_split = edges[n]
            edges_after = edges[n+1:]
            
            # get splitting point on edge
            new_pt = edge_to_split.get_pt(val)
            
            # create new edge
            new_edge = edge_to_split.copy(len(self.edges))
            
            # move points of split edge segments
            edge_to_split.replace_pt(1,new_pt)
            new_edge.replace_pt(0,new_pt)
            
            # assign new face to new edge segment
            new_edge.swap_face(target_face, new_face)
            
            # add new edge segment to edge lists of adjacent faces
            for face in edge_to_split.faces:
                if face is not target_face:
                    face.add_edge(dir[0], edge_to_split, new_edge)
            
            # remove trailing edge segments from original face
            target_face.edges[dir[0]][i] = edges[:n+1]
            
            # remove reference to original face from trailing edge segments
            for edge in edges_after:
                edge.swap_face(target_face, new_face)
            
            # add new split point and edges to list
            new_pts.append(new_pt)
            new_edges.append([new_edge] + edges_after)
            
            # add new edge to edge list
            self.edges.append(new_edge)
        
        # get reference edge in opposite direction
        ref_edge = target_face.edges[dir[1]][1]
        # iterate over all segments in reference edge
        for edge in ref_edge:
            # switch face reference to new face
            edge.swap_face(target_face, new_face)
        
        # create new edge at split and add to edges list
        splitting_edge = Edge(len(self.edges), new_pts[0], new_pts[1], [target_face, new_face])
        self.edges.append(splitting_edge)
        
        # replace edge reference in target face to new edge
        target_face.replace_edge(dir[1],1,[splitting_edge])
        
        # create edge references and assign to new face
        new_face_edges = [None,None]
        new_face_edges[dir[0]] = new_edges
        new_face_edges[dir[1]] = [[splitting_edge], ref_edge]
        new_face.set_edges(new_face_edges)
        
        # add new face to face list
        self.faces.append(new_face)
        
        return [target_face, new_face]
    
    def get_edge_geo(self, connecting_edge, min_opening):
        edges_out = []
        for i,edge in enumerate(self.edges):
            edges_out += edge.get_geo(connecting_edge[i], min_opening)
        return edges_out
    
    def get_edge_lengths(self):
        lengths_out = []
        for edge in self.edges:
            lengths_out.append(edge.get_length())
        return lengths_out
    
    def get_edge_neighbors(self):
        neighbors_out = []
        for edge in self.edges:
            neighbors_out.append([face.id for face in edge.faces])
        return neighbors_out
    
    def get_face_edges(self):
        faces_out = []
        for face in self.faces:
            faces_out.append(face.get_edges())
        return faces_out
    
    def get_face_aspects(self):
        aspects_out = []
        for face in self.faces:
            aspects_out.append(face.get_aspect())
        return aspects_out
    
    def __str__(self):
        output = ["Layout = [{}] faces, [{}] edges".format(len(self.faces), len(self.edges))]
        output.append("======")
        for face in self.faces:
            output.append(str(face))
        output.append("======")
        for edge in self.edges:
            output.append(str(edge))
        return "\n".join(output)


def get_layout(room_def, split_list, dir_list, room_order, min_opening):
    
    # load room data set and check for errors
    
    room_areas = [room["area"] for room in room_def]
    a = math.sqrt(definition["aspect"] * sum(room_areas))
    b = sum(room_areas) / a
    dims = [a,b]
    
    rooms_list = []
    
    for num in room_order:
        rooms_list.append(room_def[num])
    
    room_names = [room['name'] for room in rooms_list]

    # create subdivision tree
    ids = range(len(room_names))
    
    root = Node(ids.pop(0))
    children = root.subdivide(dir_list.pop(0), ids.pop(0))[:]
    
    terminal_nodes = []
    split_nodes = [root]
    
    for split in split_list:
        split_index = int(math.floor(len(children) * split * 0.99999999))
        
        terminal_nodes += children[:split_index]
        children = children[split_index:]
        
        split_nodes.append(children.pop(0))
        children += split_nodes[-1].subdivide(dir_list.pop(0), ids.pop(0))
    terminal_nodes += children
    
    root.collect_areas(rooms_list)
    
    split_list = [node.get_info() for node in split_nodes]

    
    room_dict = {"outside": 0}
    
    for i, room_name in enumerate(room_names):
        room_dict[room_name] = i+1
    

    adjacency_list = []
    for room in room_def:
        room1 = room_dict[room['name']]
        for room2 in room['adjacency']:
            room2 = room_dict[room2]
            if [room1, room2] not in adjacency_list and [room2, room1] not in adjacency_list:
                    adjacency_list.append([room1, room2])
    

    aspect_dict = {}
    
    for room in room_def:
        if "aspect ratio" in room.keys():
            aspect_dict[room['name']] = [True, room['aspect ratio']]
        else:
            aspect_dict[room['name']] = [False, None]
    
    aspect_bool = [aspect_dict[room][0] for room in room_names]
    
    aspect_targets = []
    for room in room_names:
        if aspect_dict[room][0]:
            aspect_targets.append(aspect_dict[room][1])
    
    
    layout = Layout([dims[0],dims[1]])
    

    for split_set in split_list:
        layout.subdivide(layout.faces[split_set[0]+1], split_set[1], split_set[2])
    
    
    zone_edges = layout.get_face_edges()[1:]
    edges_neighbors = layout.get_edge_neighbors()
    

    aspects = [a for i,a in enumerate(layout.get_face_aspects()[1:]) if aspect_bool[i]]
    
    aspect_score = 0.0
    
    for i, aspect in enumerate(aspects):
        aspect_score += abs(aspect-aspect_targets[i])
    

    
    
    edge_lengths = layout.get_edge_lengths()
    
    num_zones = len(room_names)
    
    adjacency_score = 0
    connecting_edges = []
    adjacency_violations = [0] * (num_zones + 1)
    
    for adjacency in adjacency_list:
        if adjacency in edges_neighbors and edge_lengths[edges_neighbors.index(adjacency)] >= min_opening:
            connecting_edges.append(edges_neighbors.index(adjacency))
        elif list(reversed(adjacency)) in edges_neighbors and edge_lengths[edges_neighbors.index(list(reversed(adjacency)))] >= min_opening:
            connecting_edges.append(edges_neighbors.index(list(reversed(adjacency))))
        else:
            adjacency_score += 1
            for zone_index in adjacency:
                adjacency_violations[zone_index] += 1
    
    connecting_edge = [edge in connecting_edges for edge in range(len(edges_neighbors))]
    adjacency_violations = adjacency_violations[1:]
    
    edges_out = layout.get_edge_geo(connecting_edge, min_opening)
    
    return edges_out, adjacency_score, aspect_score

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