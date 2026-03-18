from rover_navigation.planning.priority_queue import Priority, PriorityNode, PriorityQueue
import pytest

def test_priority_lt_by_k1():
    """
    test that smaller k1 wins
    """
    assert Priority(1,5) < Priority(2,0)

def test_priority_lt_by_k2_when_k1_equal():
    """
    tests that smaller k2 wins if k1 ties
    """
    assert Priority(1,3) < Priority(1,4)

def test_priority_le_equal():
    """
    tests that equal priorities behave
    """
    assert Priority(1,3) <= Priority(1,3)

def test_priority_lt_equal_is_false():
    """
    test that __lt__ is false if the keys are equal
    """
    assert not (Priority(1,3) < Priority(1,3))

def test_priority_node_lt():
    """
    test comparison
    """
    a = PriorityNode(Priority(1,2),"A")
    b = PriorityNode(Priority(1,3), "B")
    assert a < b

def test_insert_and_Top():
    """
    makes sure that smallest priortiy is on top
    """
    pq = PriorityQueue()
    pq.insert("A", Priority(5,1))
    pq.insert("B", Priority(2,7))
    pq.insert("C", Priority(2,3))
    assert pq.top() == "C"

def test_top_key_empty():
    """
    tests top key on empty queue
    """
    pq = PriorityQueue()
    key = pq.top_key()

    assert key.k1 == float("inf")
    assert key.k2 == float("inf")

def test_pop_returns_smallest():
    """
    tests that correct item is removed and heap order preserved
    """
    pq = PriorityQueue()
    
    pq.insert("A", Priority(3,0))
    pq.insert("B", Priority(1,5))
    pq.insert("C", Priority(2,1))

    item = pq.pop()
    assert item.vertex == "B"
    assert pq.top() == "C"

def test_remove_vertex():
    """
    tests that vertex is properly removed and heap still works
    """

    pq = PriorityQueue()
    pq.insert("A", Priority(1,1))
    pq.insert("B", Priority(2,2))
    pq.insert("C", Priority(3,3))

    pq.remove("A")
    assert pq.top() == "B"

def test_update():
    pq = PriorityQueue()
    pq.insert("A", Priority(5,5))
    pq.insert("B", Priority(2,2))

    pq.update("A", Priority(1,1))
    assert pq.top() == "A"