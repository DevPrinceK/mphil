import math
from typing import Optional, Dict
import random

class FibNode:
    def __init__(self, intersection_id: str, traffic_volume: float):
        self.intersection_id = intersection_id
        self.traffic_volume = traffic_volume
        self.degree = 0
        self.marked = False
        self.parent = None
        self.child = None
        self.left = self
        self.right = self

class FibonacciHeap:
    def __init__(self):
        self.min_node = None
        self.total_nodes = 0
        
    def insert(self, intersection_id: str, traffic_volume: float) -> FibNode:
        new_node = FibNode(intersection_id, traffic_volume)
        if not self.min_node:
            self.min_node = new_node
        else:
            self._add_to_root_list(new_node)
            if new_node.traffic_volume < self.min_node.traffic_volume:
                self.min_node = new_node
        self.total_nodes += 1
        return new_node
    
    def _add_to_root_list(self, node: FibNode):
        if not self.min_node:
            self.min_node = node
        else:
            node.right = self.min_node.right
            node.left = self.min_node
            self.min_node.right.left = node
            self.min_node.right = node
    
    def extract_min(self) -> Optional[FibNode]:
        z = self.min_node
        if z:
            if z.child:
                children = self._get_node_list(z.child)
                for child in children:
                    self._add_to_root_list(child)
                    child.parent = None
            
            z.left.right = z.right
            z.right.left = z.left
            
            if z == z.right:
                self.min_node = None
            else:
                self.min_node = z.right
                self._consolidate()
            
            self.total_nodes -= 1
        return z
    
    def _consolidate(self):
        max_degree = int(math.log2(self.total_nodes)) + 1
        degree_array = [None] * max_degree
        
        nodes = self._get_node_list(self.min_node)
        for node in nodes:
            degree = node.degree
            while degree_array[degree]:
                other = degree_array[degree]
                if node.traffic_volume > other.traffic_volume:
                    node, other = other, node
                self._link(other, node)
                degree_array[degree] = None
                degree += 1
            degree_array[degree] = node
        
        self.min_node = None
        for i in range(max_degree):
            if degree_array[i]:
                if not self.min_node:
                    self.min_node = degree_array[i]
                else:
                    self._add_to_root_list(degree_array[i])
                    if degree_array[i].traffic_volume < self.min_node.traffic_volume:
                        self.min_node = degree_array[i]

    def _link(self, child: FibNode, parent: FibNode):
        child.left.right = child.right
        child.right.left = child.left
        child.parent = parent
        if not parent.child:
            parent.child = child
            child.right = child
            child.left = child
        else:
            child.right = parent.child.right
            child.left = parent.child
            parent.child.right.left = child
            parent.child.right = child
        parent.degree += 1
        child.marked = False

    def _get_node_list(self, start: FibNode) -> list:
        if not start:
            return []
        nodes = []
        current = start
        while True:
            nodes.append(current)
            current = current.right
            if current == start:
                break
        return nodes

class TrafficManagementSystem:
    def __init__(self):
        self.heap = FibonacciHeap()
        self.intersection_nodes: Dict[str, FibNode] = {}
        
    def add_intersection(self, intersection_id: str, initial_volume: float):
        """Add a new intersection to the system"""
        node = self.heap.insert(intersection_id, initial_volume)
        self.intersection_nodes[intersection_id] = node
        
    def get_critical_intersection(self) -> Optional[str]:
        """Get the intersection with highest traffic volume"""
        node = self.heap.extract_min()
        if node:
            return node.intersection_id
        return None
    
    def optimize_traffic_signals(self):
        """Optimize traffic signals based on traffic volumes"""
        while True:
            critical_intersection = self.get_critical_intersection()
            if not critical_intersection:
                break
            
            # Simulate traffic signal optimization for the critical intersection
            print(f"Optimizing signals at intersection {critical_intersection}")
            
            # In a real system, this would:
            # 1. Adjust green light duration
            # 2. Update adjacent intersection timings
            # 3. Monitor traffic flow changes
            
def simulate_traffic_management():
    # Create traffic management system
    tms = TrafficManagementSystem()
    
    # Add intersections with initial traffic volumes
    # Lower values indicate higher congestion (priority)
    # generate intersections from predefined lists
    pre = ['GCB', 'NNB', 'CS', 'ITD', 'Akuafo', 'Sarbah', 'Legon', 'Pent']
    post = ['road', 'round about', 'junction', 'interchange']

    # Ask user to enter number of intersections to simulate.
    num = input('Enter Number of Intersections to Simulate: ')
    num = int(num)

    intersections = {
        f"{random.choice(pre)}-{random.choice(post)}-{i}" :  round(random.uniform(1, 5), 1) for i in range(0, num)
    }

    
    for intersection_id, volume in intersections.items():
        tms.add_intersection(intersection_id, volume)
        print(f"Added intersection {intersection_id} with traffic volume {volume}")
    
    print("\nOptimizing traffic signals for critical intersections:")
    tms.optimize_traffic_signals()

if __name__ == "__main__":
    simulate_traffic_management()