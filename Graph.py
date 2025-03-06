import heapq

class Graph:
    def __init__(self, V: int):
        self.V = V
        self.adj = [[] for _ in range(V)]

    def add_edge(self, a: int, b: int, w: float, id: int):
        self.adj[a].append((b, w, id))
    
    def clear_edges(self):
        self.adj = [[] for _ in range(self.V)]
    
    def dynamic_dijkstra(self, src: int, target: int) -> tuple[float, list, list]:
        pq = []
        heapq.heappush(pq, (0, src))

        dist = [float('inf')] * self.V
        dist[src] = 0

        prev = [None] * self.V
        edges = [None] * self.V

        while pq:
            d, node = heapq.heappop(pq)

            for v, w, id in self.adj[node]:
                if dist[v] > d + w:
                    dist[v] = d + w
                    prev[v] = node
                    edges[v] = id
                    heapq.heappush(pq, (dist[v], v))

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

        return dist[target], path, edges_used