import pandas as pd
from Graph import Graph
from DataFetch import get_dataframe_content

'''
In order to predict the route in a reasonable time, it's going to be necessary to fetch
some real-time data.
Because the model's LOOKBACK = 24, it's going to be necessary at least 24 timestamps for each segment
at least 24 * 1047 = 25128 rows
Fetching data takes a long time, in this case, we're going to use a recursive approach to predict next values
and avoid refetching
Gotta make easier to map and remap values too
'''

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

def run():

    content = get_dataframe_content(10)
    print(content.head())

    vertices_compress, vertices_map, streets_set = get_sets(content)
    print(f"Endpoints: {len(vertices_map)}")
    print(f"Segments: {len(streets_set)}")

    city = create_city_graph(vertices_compress, vertices_map, streets_set)

    SRC = "Damen"
    TARGET = "Western"
    MAX_SPEED = 60 # Km

    sssp = city.dynamic_dijkstra(vertices_compress[SRC], vertices_compress[TARGET])
    HAS_ROUTE = (sssp[0] != float('inf'))

    print(f"\n < {SRC} ==> {TARGET} >")
    print(f" Max. Velocity: {MAX_SPEED} km/h\t Time: " + str(sssp[0]) + " seconds   Distance: " + str(sssp[1]) + " km")
    if HAS_ROUTE: print(f" The shortest route is: \n {[vertices_map[node] for node in sssp[2]]} \n {sssp[3]}")
    else: print(f" There's no route between {SRC} and {TARGET}")

run()