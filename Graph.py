import heapq
from NeuralNetwork import NeuralNetwork
from DataFetch import get_newest_test_sample

MAX_SPEED = 60
class Graph:
    def __init__(self, V: int):
        self.V = V
        self.adj = [[] for _ in range(V)]

    def add_edge(self, a: int, b: int, w: float, id: int):
        self.adj[a].append((b, w, id))
    
    def clear_edges(self):
        self.adj = [[] for _ in range(self.V)]
    
    def dynamic_dijkstra(self, src: int, target: int) -> tuple[float, float, list, list]:
        pq = []
        heapq.heappush(pq, (0, 0, src))

        dist = [float('inf')] * self.V
        dist[src] = 0

        time = [float('inf')] * self.V
        time[src] = 0

        prev = [None] * self.V
        edges = [None] * self.V

        nn = NeuralNetwork("Model-10x-20250308-003101-AC-55.h5")
        x, y, compressed_segments = get_newest_test_sample(MAX_SPEED)
        t_y = y.transpose()
        print(x.shape)
        print(y.shape)

        while pq:
            t, d, node = heapq.heappop(pq)

            for v, w, id in self.adj[node]:
                # For each edge segment, predict its traffic speed to accurately measue the time
                p = nn.predict(x, t_y[compressed_segments[id]], 1, plot=False)
                #print(p)
                speed = p[-1]*MAX_SPEED
                if time[v] <= t + w/speed*3600: continue
                time[v] = t + w/speed*3600
                dist[v] = d + w
                prev[v] = node
                edges[v] = id
                heapq.heappush(pq, (time[v], dist[v], v))

        if dist[target] == float('inf'): return float('inf'), [], []

        path = []
        edges_used = []
        current = target
        while current is not None:
            path.append(current)
            edges_used.append(edges[current])
            current = prev[current]
        path.reverse()

        edges_used.pop()
        edges_used.reverse()

        return time[target], dist[target], path, edges_used