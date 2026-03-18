"""
g(s) - current best known path cost from the start to s
rhs(s) - one-step lookahead cost 

algorithim wants g(s) = rhs(s)

if g(s) != rhs(s), must process the states
- for deciding which states each state gets two keys
- h(start,s) <-- heuristic 
- k_m <-- running modifier

start with smallest k1 (state that is closest to the goal),
if k1 ties, smallest k2 (raw best known cost)

three cases
(1) g(s) > rhs(s)
- overconsistent (g too high), decrease g to match rhs
(2) g(s) < rhs(s) 
- (g too low), increase g
(3) g(s) = rhs(s)
- desired condition, don't need to process

lexicographic: (a1,a2) < (b1,b2) if a1 > b1 or a1 = b1 and a2 < b2
"""


class Priority:
    """
    handle order of keys
    """
    def __init__(self, k1, k2):
        """
        :param k1: key values
        :param k2: key values
        """
        self.k1 = k1
        self.k2 = k2

    
    def __lt__(self, other):
        """
        lexiographic 'lower than'
        :param other: comparable key
        :return: lexiographic order
        """
        return self.k1 < other.k1 or (self.k1 == other.k1 and self.k2 < other.k2)
    
    def __le__(self, other):
        """
        lexiographic 'lower than or equal'
        :param other: comparble keys
        :return: lexicographic order
        """
        return self.k1 < other.k1 or (self.k1 == other.k1 and self.k2 <= other.k2)
    
class PriorityNode:
    """
    lexicographich order of vertices
    """

    def __init__(self, priority, vertex):
        """
        :param priority: priority of vertex
        :param vertex: 
        """
        self.priority = priority
        self.vertex = vertex

    def __le__(self, other):
        """
        :param other: comparable node
        :return: lexicographic order
        """
        return self.priority <= other.priority
    
    def __lt__(self,other):
        """ 
        :param other: comparible node
        :return: lexicographic order
        """
        return self.priority < other.priority


class PriorityQueue:
    def __init__(self):
        self.heap = []
        self.vertices_in_heap = []

    # get the vertex with the lowest priority
    def top(self):
        return self.heap[0].vertex

    # return the smallest key, if its empty return inf
    def top_key(self):
        if len(self.heap) == 0: return Priority(float('inf'), float('inf'))
        return self.heap[0].priority

    
    def pop(self):
        """ 
        pop smallest from heap, maintain heap invariant
        """
        lastelt = self.heap.pop()  # raises appropriate IndexError if heap is empty
        self.vertices_in_heap.remove(lastelt.vertex)
        if self.heap:
            returnitem = self.heap[0]
            self.heap[0] = lastelt
            self._siftup(0)
        else:
            returnitem = lastelt

        return returnitem

    # heap insertion
    def insert(self, vertex, priority):
        """
        :param self: 
        :param vertex: 
        :param priority:
        """
        item = PriorityNode(priority, vertex)
        self.vertices_in_heap.append(vertex)
        self.heap.append(item)
        self._siftdown(0,len(self.heap) - 1)

    # heap removal
    def remove(self, vertex):
        self.vertices_in_heap.remove(vertex)
        for index, priority_node in enumerate(self.heap):
            if priority_node.vertex == vertex:
                self.heap[index] = self.heap[len(self.heap) - 1]
                self.heap.remove(self.heap[len(self.heap) - 1])
                break
        self.build_heap()
    
    # update the vertex in priority queue
    def update(self, vertex, priority):
        for index, priority_node in enumerate(self.heap):
            if priority_node.vertex == vertex:
                self.heap[index].priority = priority
                break
        self.build_heap()

    
    def build_heap(self):
        """Transform list into a heap, in-place, in O(len(x)) time."""
        n = len(self.heap)
        # Transform bottom-up.  The largest index there's any point to looking at
        # is the largest with a child index in-range, so must have 2*i + 1 < n,
        # or i < (n-1)/2.  If n is even = 2*j, this is (2*j-1)/2 = j-1/2 so
        # j-1 is the largest, which is n//2 - 1.  If n is odd = 2*j+1, this is
        # (2*j+1-1)/2 = j so j-1 is the largest, and that's again n//2-1.
        for i in reversed(range(n // 2)):
            self._siftup(i)
    
    def _siftdown(self, startpos, pos):
        newitem = self.heap[pos]
        # Follow the path to the root, moving parents down until finding a place
        # newitem fits.
        while pos > startpos:
            parentpos = (pos - 1) >> 1
            parent = self.heap[parentpos]
            if newitem < parent:
                self.heap[pos] = parent
                pos = parentpos
                continue
            break
        self.heap[pos] = newitem

    def _siftup(self, pos):
        endpos = len(self.heap)
        startpos = pos
        newitem = self.heap[pos]
        # Bubble up the smaller child until hitting a leaf.
        childpos = 2 * pos + 1  # leftmost child position
        while childpos < endpos:
            # Set childpos to index of smaller child.
            rightpos = childpos + 1
            if rightpos < endpos and not self.heap[childpos] < self.heap[rightpos]:
                childpos = rightpos
            # Move the smaller child up.
            self.heap[pos] = self.heap[childpos]
            pos = childpos
            childpos = 2 * pos + 1
        # The leaf at pos is empty now.  Put newitem there, and bubble it up
        # to its final resting place (by sifting its parents down).
        self.heap[pos] = newitem
        self._siftdown(startpos, pos)