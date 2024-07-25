"""
Utilities for tree traversal coordinations.

Also see README.md section "Tree Traversal Coordinations".
"""

import heapq
from collections import Counter
from threading import Lock
from typing import Dict, Optional, Tuple

TreeID = int
NodeID = str
NodeTuple = Tuple[TreeID, NodeID]


class ForestTraversalCoordinator:
    """
    Implementation of tree-traversal coordinator
    as portrayed in README.md. Thread-safe.

    Prioritize finish building trees with smaller tree_id
    before continuing with other trees.

    - add_node
    - get_node


    Params:
        Initial_nodes: dictionary mapping (tree_name, node_name)
        to num_children, one for each initial node.
    """

    def __init__(self, initial_nodes: Dict[NodeTuple, int] = {}):
        self.threading_lock = Lock()

        # Initialize counter for tracking number of times
        # a particular (tree_id, node_id) pair was returned
        # in get_node.
        self.presentation_counter = Counter()

        # Dictionary storing the maximum number of times
        # for a particular node to be returned in get_node.
        # Maps (tree_id, node_id) to a positive integer.
        self.max_presentations: Dict[Tuple[TreeID, NodeID], int] = {}

        # Initialize double-ended queue of (tree_id, node_id)
        # for tracking nodes capable of having children.
        # Skip node tuples with a zero max_presentation value.
        self.node_heapq = []
        for node_tuple, max_presentation in initial_nodes.items():
            if max_presentation > 0:
                self.max_presentations[node_tuple] = max_presentation
                self.node_heapq.append(node_tuple)

        heapq.heapify(self.node_heapq)

    def add_node(self, node_tuple: NodeTuple, max_children: int):
        """
        Add a root node to the forest.
        This implementation is thread-safe.

        A node is not added if max_children is 0.

        Params:
            (tree_id, node_id), where:
                tree_id: to enable depth-first retrieval.
                node_id: unique within the given tree.

            max_children: int, number of children of this node.
        """
        if max_children == 0:
            return

        with self.threading_lock:
            # Add this root node to the right side of the deque.
            self.max_presentations[node_tuple] = max_children
            heapq.heappush(self.node_heapq, node_tuple)

    def get_node(self) -> Optional[Tuple[TreeID, NodeID]]:
        """
        Retrieve a node from the forest, given that the
        total number of times this node is returned
        does not exceed self.max_width.

        Whenever available, prioritize nodes from trees
        with the least tree_id values.

        Important: if no node is available at the moment,
        return None immediately. The caller is responsible
        for checking again at another time.

        This method is thread safe.

        Returns:
            tree_id: str, node_id: str
        """
        with self.threading_lock:
            # Return None if no node is available.
            if len(self.node_heapq) == 0:
                return None

            # Retrieve the node tuple at the top of the heap
            # without modifying the heap at the moment.
            node_tuple = self.node_heapq[0]

            # Add 1 to the counter.
            self.presentation_counter[node_tuple] += 1

            # If the counter value of this tuple is now equal
            # to max_width, delete this tuple from both the
            # heap queue and the counter.
            num_presentations = self.presentation_counter[node_tuple]
            max_presentations = self.max_presentations[node_tuple]

            if num_presentations >= max_presentations:
                heapq.heappop(self.node_heapq)
                self.presentation_counter.pop(node_tuple)
                self.max_presentations.pop(node_tuple)

            return node_tuple
