import pandas as pd
from Graph import Graph
from DataFetch import get_dataframe_content
import pickle
import os.path
import sys

def get_sets(content: pd.DataFrame):
    vertices_compress = {}
    vertices_map = {}
    streets_set: set = set()
    for index, row in content.iterrows():
        #print(streets)
        vertices_compress[row["from_street"]] = 0
        vertices_compress[row["to_street"]] = 0
        streets_set.add((row["segment_id"],
                        #row["hour"],
                        #row["day_of_week"],
                        row["from_street"],
                        row["to_street"],
                        row["length"],
                        #row["speed"]
                        ))
    
    # Compress
    vertices_compress = {key: i + 1 for i, key in enumerate(vertices_compress)}
    vertices_map = {(i+1): st_name for i, st_name in  enumerate(vertices_compress)}
    
    #print(vertices_compress[content.at[150, "from_street"]])
    #print(vertices_map[vertices_compress[content.at[150, "from_street"]]])

    return vertices_compress, vertices_map, streets_set


# TODO: Because the city graph is always the same, it makes sense to save it into a binary file 
def create_city_graph(vertices_compress, vertices_map, streets_set) -> Graph:
    VERTICES_THRESHOLD = 150

    #c = 0
    city = Graph(len(vertices_map)+1)
    for segment in streets_set:
        if vertices_compress[segment[1]] > VERTICES_THRESHOLD and vertices_compress[segment[2]] > VERTICES_THRESHOLD: continue # Limit number of nodes
        city.add_edge(vertices_compress[segment[1]], vertices_compress[segment[2]], segment[3], segment[0])
        #c+=1
        #print(f"{segment[0]} e({segment[1]}, {segment[2]}, {segment[3]})")
    #print("edge count: " + str(c))
    return city

def get_city_graph(vertices_compress, vertices_map, streets_set) -> Graph:
    FILEPATH = "city-graph.bin"
    if os.path.isfile(FILEPATH): return pickle.load(open(FILEPATH, "rb"))

    city = create_city_graph(vertices_compress, vertices_map, streets_set)
    pickle.dump(city, open(FILEPATH, "wb"))
    return city


MAX_SPEED = 60 # Km
def find_routes(city: Graph, SRC: str, TARGET: str, vertices_compress, vertices_map, save_output: bool = True):
    FILEPATH = f"Logs/{SRC}-{TARGET}_log.txt"
    logs = open(FILEPATH, 'w')
    outputfile = logs if save_output else sys.stdout

    sssp = city.dijkstra(vertices_compress[SRC], vertices_compress[TARGET])
    HAS_ROUTE = (sssp[0] != float('inf'))

    print(f"\n < {SRC} ==> {TARGET} >", file=outputfile)
    print(f" Max. Velocity: {MAX_SPEED} km/h\t Time: " + str(sssp[0]) + " seconds   Distance: " + str(sssp[1]) + " km" , file=outputfile)
    if HAS_ROUTE: print(f" The shortest route is: \n {[vertices_map[node] for node in sssp[2]]} \n {sssp[3]}", file=outputfile)
    else: print(f" There's no route between {SRC} and {TARGET}", file=outputfile); 


    ssfp = city.dijkstra_time(vertices_compress[SRC], vertices_compress[TARGET])
    HAS_ROUTE = (ssfp[0] != float('inf'))

    print(f"\n < {SRC} ==> {TARGET} >", file=outputfile)
    print(f" Max. Velocity: {MAX_SPEED} km/h\t Time: " + str(ssfp[0]) + " seconds   Distance: " + str(ssfp[1]) + " km", file=outputfile)
    if HAS_ROUTE: print(f" The fastest route is: \n {[vertices_map[node] for node in ssfp[2]]} \n {ssfp[3]}", file=outputfile)
    else: print(f" There's no route between {SRC} and {TARGET}", file=outputfile)

    print("SSFP is %.2f seconds faster than normal SSSP" % (city.get_path_time(sssp[3])-ssfp[0]), file=outputfile)

    ssfp = city.dynamic_dijkstra(vertices_compress[SRC], vertices_compress[TARGET])
    HAS_ROUTE = (ssfp[0] != float('inf'))

    print(f"\n < {SRC} ==> {TARGET} >", file=outputfile)
    print(f" Max. Velocity: {MAX_SPEED} km/h\t Time: " + str(ssfp[0]) + " seconds   Distance: " + str(ssfp[1]) + " km", file=outputfile)
    if HAS_ROUTE: print(f" The fastest future route is: \n {[vertices_map[node] for node in ssfp[2]]} \n {ssfp[3]}", file=outputfile)
    else: print(f" There's no route between {SRC} and {TARGET}", file=outputfile)

    logs.close()

def run():

    content = get_dataframe_content(10)
    print(content.head())

    vertices_compress, vertices_map, streets_set = get_sets(content)
    print(f"Endpoints: {len(vertices_map)}")
    print(f"Segments: {len(streets_set)}")

    SRC = "Mendota"
    TARGET = "Oriole"

    city = create_city_graph(vertices_compress, vertices_map, streets_set)

    find_routes(city, SRC, TARGET, vertices_compress, vertices_map)

run()