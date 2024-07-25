"""
Utils related to tree structures for storing reasoning traces.
- Serialize a tree into a JSON-serializable dictionary of lists
- Load a tree from a dictionary of lists.
- Iterating through the tree, returning a specific number of 
    child elements at each node, along with all nodes along the 
    path from root node to this node.
    - Return only one child element at a time for rollout generation
    - Return a pair of child elements for contrastive feedback
"""

import math
from threading import Lock
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    NamedTuple,
    Optional,
    Tuple,
    Union,
)

from .threading_utils import Cache
from .traversal_utils import ForestTraversalCoordinator, NodeTuple

Attributes = Dict[str, Optional[Union[str, bool, int]]]

ForestName = str
JobName = str
TreeId = int
NodeId = str
Prediction = float

ReasoningTreeIdTuple = Tuple[ForestName, TreeId]

ReasoningTreeFields = Literal["nodes", "edges", "attributes"]


class _ConfidenceInterval(NamedTuple):
    """
    Example: a p-value of 0.05 is for 95% confidence intervals.
    """

    lower_bound: float
    mean: float
    upper_bound: float
    p: float = 0.05


class ReasoningTreeNode(NamedTuple):
    """
    Provides structural requirements on each node of the
    reasoning tree.

    Node ID within the tree is intentionally not included,
    since node_id is relevant only from the perspective of the
    tree (a consumer of the tree node class) and should not
    be included here so as to minimize dependency.
    """

    attributes: Dict[str, Any] = {}
    data: Dict[str, Any] = {}

    def to_dict(self) -> Dict:
        """
        Generate JSON-friendly dictionary representation of self.
        """
        serializable_payload = {"attributes": self.attributes, "data": self.data}
        return serializable_payload

    @staticmethod
    def from_dict(payload: Dict) -> "ReasoningTreeNode":
        """
        Parse dictionary to create an instance of self and
        verify data format.
        """

        assert "attributes" in payload.keys()
        assert "data" in payload.keys()

        return ReasoningTreeNode(payload["attributes"], payload["data"])

    @property
    def value_estimate(self) -> Optional[_ConfidenceInterval]:
        """
        Estimation of the value of this node based on the subtree
        below this node.

        Returns:
            None if num_leaf_nodes attribute is 0 or not yet set.
            Otherwise, returns an instance of _ConfidenceInterval.
        """
        num_leaf_nodes = self.attributes.get("num_leaf_nodes")
        leaf_value_sum = self.attributes.get("leaf_value_sum")

        if (num_leaf_nodes is None) or (leaf_value_sum is None):
            return None

        return get_value_estimate(num_leaf_nodes, leaf_value_sum)


def invert_edges(edges: Dict[str, List[str]]) -> Dict[str, str]:
    """
    Invert parent to children map to get a parent map.

    Params:
        edges: Dict[str, List[str]] map parent to children.

    Returns:
        Dictionary where key is child node id and value is parent id.
    """
    inverted_edges: Dict[str, str] = {}
    for parent_node_id, children in edges.items():
        for child_node_id in children:
            inverted_edges[child_node_id] = parent_node_id

    return inverted_edges


def get_next_node_id(node_identifiers: List[str]) -> str:
    """
    Given a list of node identifiers that are already assigned,
    return the next-available node identifier.

    Params:
        node_identifiers: list of node identifiers.

    Returns:
        node identifier string.
    """
    max_node_identifier_number: int = -1
    for node_identifier in node_identifiers:
        try:
            node_identifier_number = int(node_identifier)
        except ValueError:
            continue

        max_node_identifier_number = max(
            node_identifier_number, max_node_identifier_number
        )

    return str(max_node_identifier_number + 1)


def get_altitude_map(edges: Dict[NodeId, List[NodeId]]) -> Dict[NodeId, int]:
    """
    Return a dictionary mapping node (node_id)
    to the length of a path from root to this node.
    The root node would have an "altitude" of 1.

    Returns:
        Dict mapping node_id to altitude.
    """
    # Idea: breadth-first search from the root node.
    nodes_to_visit: List[NodeId] = ["0"]
    altitude_map: Dict[NodeId, int] = {"0": 1}

    # For each node, update the depth of its children
    # add place all its children on the list
    # to visit next.
    while len(nodes_to_visit) > 0:
        parent_node_id = nodes_to_visit.pop(-1)
        parent_depth = altitude_map[parent_node_id]

        # Nodes that don't have any children might
        # not be in the edge map.
        children_ids = edges.get(parent_node_id, [])

        for child_id in children_ids:
            if child_id in altitude_map.keys():
                raise ValueError("Each node should be visited only once")

            altitude_map[child_id] = parent_depth + 1
            nodes_to_visit.append(child_id)

    return altitude_map


def get_value_estimate(
    num_leaf_nodes: int, leaf_value_sum: int
) -> Optional[_ConfidenceInterval]:
    """
    Compute "value" estimate of the given node.

    Params:
        num_leaf_nodes: number of leaf nodes under this node.
            1 if this node itself is a leaf node.
        leaf_value_sum: sum of value of leaf nodes under this node.
            If this node is a leaf node, leaf_value_sum is 1 if
            the leaf prediction is correct, and 0 otherwise.

    Returns:
        None if num_leaf_nodes is zero or negative.
        Confidence interval assuming a binomial distribution.

    """
    p = 0.05
    z_score = 1.96

    if num_leaf_nodes <= 0:
        return None

    mean_value = leaf_value_sum / num_leaf_nodes
    standard_error = math.sqrt(mean_value * (1 - mean_value) / num_leaf_nodes)
    return _ConfidenceInterval(
        lower_bound=mean_value - z_score * standard_error,
        mean=mean_value,
        upper_bound=mean_value + z_score * standard_error,
        p=p,
    )


class PathElement(NamedTuple):
    """
    Represents one of the many steps that make a path in
    a reasoning tree.

    Important: self.node must be an element of self.siblings.
    """

    node_id: NodeId
    node: ReasoningTreeNode
    siblings: List[ReasoningTreeNode]

    @staticmethod
    def serialize_path(path: List["PathElement"]) -> List[Dict[str, Any]]:
        """
        Produce a JSON-serializable representation of the given
        "path"- a list of path elements.

        Params:
            path: List[PathElement], list of path elements.

        Returns:
            List[Dict[str, Any]]: JSON-serializable path.
        """
        output: List[Dict[str, Any]] = []

        for node_id, node, siblings in path:
            # Crete serialized representation of this rollout step.
            step_representation = {
                "node_id": node_id,
                "attributes": node.attributes,
                "data": node.data,
                "siblings": [sibling.to_dict() for sibling in siblings],
            }

            output.append(step_representation)

        return output

    @staticmethod
    def load_path(path_serialized: List[Dict[str, Any]]) -> List["PathElement"]:
        """
        TODO: De-serialize a given path. This method would simplify the
        parsing logic in rollout worker and AI preference generation.

        Params:
            path_serialized: List[Dict[str, Any]], from `serialize_path`
            (also see above).

        Returns:
            List["PathElement"], a "path"- a list of path elements.
        """
        raise NotImplementedError()


class ReasoningTree:
    ROOT_NODE_ID: NodeId = "0"

    def __init__(self, attributes=None, nodes=None, edges=None):
        self.attributes: Dict[str, Any] = attributes if attributes is not None else {}

        # Store nodes and edges separately.
        # Map NodeID to node items.
        self.nodes: Dict[str, ReasoningTreeNode] = nodes if nodes is not None else {}

        # Map NodeID of a node to NodeID of its children.
        self.edges: Dict[str, List[str]] = edges if edges is not None else {}

        # Map a node to its predecessor
        self.edges_inverted = invert_edges(self.edges)

        # Map a node to the number of nodes in a path
        # from root to this node.
        self.altitude_map = get_altitude_map(self.edges)

        # Insert altitude info into each node.
        # Refer to Engineering considerations:
        # limiting maximum tree height for branching
        for node_id in self.nodes.keys():
            self.nodes[node_id].attributes["altitude"] = self.altitude_map[node_id]

        self.threading_lock = Lock()

    def is_leaf(self, node_id: NodeId) -> bool:
        """
        Return whether a node has any children node.

        Params:
            node_id: NodeId of the node to query.

        Returns:
            boolean. True if the node has no children.
        """
        children_node_ids = self.edges.get(node_id, [])
        num_children = len(children_node_ids)

        return num_children == 0

    def to_dict(self) -> Dict[ReasoningTreeFields, Dict]:
        """
        Return JSON-serializable dictionary representation of the tree.

        Returns:
            Dict[str, Any] with keys "nodes", "edges", and "attributes"
        """
        # Represent each node as a JSON-friendly dictionary
        serialized_nodes = {key: node.to_dict() for key, node in self.nodes.items()}

        serializable_payload: Dict[ReasoningTreeFields, Dict] = {
            "nodes": serialized_nodes,
            "edges": self.edges,
            "attributes": self.attributes,
        }

        return serializable_payload

    @staticmethod
    def from_dict(payload: Dict[str, Dict]) -> "ReasoningTree":
        """
        Create reasoning tree from dictionary,
        where the format of the dictionary should be similar
        to the output of the to_dict method.

        Apply data validation.

        Params:
            payload: Dict[str, Any] with keys "nodes", "edges", and "attributes"

        Returns:
            Reasoning Tree instance.
        """
        assert "nodes" in payload.keys()
        assert "edges" in payload.keys()
        assert "attributes" in payload.keys()

        # De-serialize ReasoningTreeNode instances in nodes.
        nodes_raw = payload["nodes"]
        nodes = {
            key: ReasoningTreeNode.from_dict(node_dict)
            for key, node_dict in nodes_raw.items()
        }

        return ReasoningTree(
            attributes=payload["attributes"], nodes=nodes, edges=payload["edges"]
        )

    def add_node(self, node: ReasoningTreeNode, parent_id: NodeId) -> NodeId:
        """
        Add node to reasoning tree, returning assigned node id.

        This implementation is thread-safe.

        Params:
            node: ReasoningTreeNode to add to the tree.
            parent_id: NodeId, id of parent node.

        Returns:
            node_id.
        """
        with self.threading_lock:
            node_id = get_next_node_id(list(self.nodes.keys()))
            node.attributes["node_id"] = node_id

            altitude = self.altitude_map[parent_id] + 1
            if "altitude" in node.attributes.keys():
                assert node.attributes["altitude"] == altitude, (node, altitude)

            node.attributes["altitude"] = altitude

            self.nodes[node_id] = node
            self.altitude_map[node_id] = altitude

            sibling_list = self.edges.get(parent_id, [])
            sibling_list.append(node_id)
            self.edges[parent_id] = sibling_list
            self.edges_inverted[node_id] = parent_id

        return node_id

    def get_sibling_ids(self, node_id: NodeId) -> List[NodeId]:
        """
        Return the node_id of all nodes with the same parent as
        the given node. Important: The node_id of the selected node
        itself would also be included.

        Params:
            node_id: NodeId of the node to look up.

        Returns:
            List[NodeId]: List of node_ids, one for each sibling
                of this node, as well as this node itself.
        """
        # Root node has no siblings.
        # However, by definition, the root node still needs to be
        # included in the sibling list.
        if node_id == self.ROOT_NODE_ID:
            return [node_id]

        parent_id = self.edges_inverted[node_id]
        sibling_ids = self.edges.get(parent_id, [])

        return sibling_ids

    def get_siblings(self, node_id: NodeId) -> List[ReasoningTreeNode]:
        """
        Wrapper around get_siblings_ids,
        returning reasoning tree node instances instead of node_ids.
        """
        siblings = []
        for sibling_id in self.get_sibling_ids(node_id):
            sibling_node = self.nodes.get(sibling_id)
            if sibling_node is not None:
                siblings.append(sibling_node)

        return siblings

    def get_walk(self, node_id: NodeId) -> List[PathElement]:
        """
        Return a walk from the root of the tree to a given node.

        Params:
            node_id: node to walk to.

        Returns:
            List of reasoning tree (node_id, nodes, siblings) between
            root node and the given node, inclusive, in the order from
            root to the given node.

            Raises KeyError if node_id is not found.
        """
        if node_id not in self.nodes.keys():
            raise KeyError(
                "node_id {} is not found in tree. Options: {}".format(
                    node_id, list(self.nodes.keys())
                )
            )

        # Create list of nodes visited in inverted order.
        path_inverted: List[PathElement] = []
        current_node_id = node_id

        # Limit number of iterations to tree depth to prevent infinite loop.
        for _ in range(len(self.nodes.keys())):
            # Add self to the node list.

            # Concurrency considerations: if a thread added a
            # node to the tree, the result would be the same.
            # Hence, there is no need to set a lock around this loop.
            path_element = PathElement(
                node_id=current_node_id,
                node=self.nodes[current_node_id],
                siblings=self.get_siblings(current_node_id),
            )
            path_inverted.append(path_element)

            parent_id = self.edges_inverted.get(current_node_id)

            if parent_id is None:
                break

            current_node_id = parent_id

        # Return path, inverted
        return path_inverted[::-1]

    def _select_child_deterministic(self, parent_id: NodeId) -> Optional[NodeId]:
        """
        Among all the child nodes of the given node,
        deterministically select one of these children
        and return the node id of that child node.

        See design note of get_walk_to_leaf on the
        intended "determinism" of this method.

        Params:
            parent_id: NodeId of parent node.

        Returns:
            node_id of the selected child,
            or None if the given "parent" node is a leaf node.
        """
        children_ids = self.edges.get(parent_id, [])
        if len(children_ids) == 0:
            return None

        # Try converting each child node id into an int
        # and return the minimum value, or the first value
        # that isn't convertible.
        child_node_id_int_values: List[int] = []
        for child_node_id_str in children_ids:
            try:
                child_node_id_int = int(child_node_id_str)
            except ValueError:
                return child_node_id_str

            child_node_id_int_values.append(child_node_id_int)

        child_node_id_selected: NodeId = str(min(child_node_id_int_values))
        return child_node_id_selected

    def get_walk_to_leaf(self, node_id: NodeId) -> List[PathElement]:
        """
        Return a path starting from root, reaches the given node,
        and continues until reaching a leaf node.

        Design note: while the path between root and the selected
        node is unique, multiple leaf nodes might separate from the
        selected node. This method will ensure that as long as the
        subtree beneath the givennode stays the same, the leaf that
        is eventually reached in the path returned from this method
        would also be the same.

        Note that if the given node is the root node, that node
        will be included only at the beginning of the path returned
        and will not be repeated.

        Params:
            node_id: NodeId, of the node that the path should traverse
                through.

        Returns:
            List[PathElement], refer to PathElement for specifications.

        - Retrieve path from root to the selected node.
        - Set current node to the selected node.
        - Loop for up to num_nodes times.
            - (The current node should already be in the path)
            - Select a child and create a PathElement for that child node.
                - If no child is found (current node is a leaf,) stop the loop.
            - Add the path element of the child node to the path
            - Set child node as the new "current node"
        """

        # Use the path from root to the selected node as starting point.
        path: List[PathElement] = self.get_walk(node_id)
        assert len(path) > 0

        # The current node should already be in the path
        current_node_id = path[-1].node_id

        # Use a for loop instead of a while loop to
        # avoid infinite loops.
        for _ in range(len(self.nodes.keys())):
            child_id_selected = self._select_child_deterministic(current_node_id)

            if child_id_selected is None:
                break

            child_path_element = PathElement(
                node_id=child_id_selected,
                node=self.nodes[child_id_selected],
                siblings=self.get_siblings(child_id_selected),
            )
            path.append(child_path_element)
            current_node_id = child_id_selected

        return path

    def update_branch_values(self, leaf_node_id: NodeId):
        """
        After a branch is added, invoke this method
        on the new leaf node of that branch to update
        the value of all nodes between root and the
        given leaf node.

        Params:
            leaf_node_id: NodeId of the leaf node
                from the branch.

        Overview:
        - Get path from root to the leaf node
        - Update value attributes each node along the path,
            starting from leaf
            - Use default value if the attribute is not yet
              initialized.
        """
        # Retrieve the "value" of the leaf node.
        leaf_node = self.nodes[leaf_node_id]
        leaf_value = leaf_node.attributes["leaf_value_sum"]

        # Iteratively update (in-place) value
        # attributes of all nodes along the path
        # from root to this leaf, starting from the node
        # right above leaf.
        path_to_leaf = self.get_walk(leaf_node_id)
        for path_element in path_to_leaf[:-1]:
            node_attributes = path_element.node.attributes

            with self.threading_lock:
                num_leaf_nodes_initial = node_attributes.get("num_leaf_nodes", 0)
                leaf_value_sum_initial = node_attributes.get("leaf_value_sum", 0)
                node_attributes["num_leaf_nodes"] = num_leaf_nodes_initial + 1
                node_attributes["leaf_value_sum"] = leaf_value_sum_initial + leaf_value

    def get_leaf_node_ids(self) -> List[NodeId]:
        """
        Returns:
            list of node_ids of all leaf nodes in this tree.
        """
        output: List[NodeId] = list(filter(self.is_leaf, self.nodes.keys()))
        return output


# Signature for method that returns number of children given a tree node.
GetNumChildrenMethod = Callable[[ReasoningTreeNode], int]

# Signature for method that retrieves tree from database.
GetTreeMethod = Callable[[ReasoningTreeIdTuple], Optional[ReasoningTree]]

# Signature for method that retrieves all reasoning trees from the forest.
GetForestMethod = Callable[[ForestName], Optional[List[ReasoningTree]]]

# Signature for method that uploads tree to database.
UploadTreeMethod = Callable[[ReasoningTreeIdTuple, ReasoningTree], None]


class ForestTraversalManager:
    """
    Thread-safe abstraction of the following methods:
    - get_node(forest_name, job_name)
    - add_node(forest_name, job_name, tree_id, parent_id, max_children)
        - Add node to traversal manager.
        - Add node to tree.
        - Upload tree to database using callback method specified when
        instantiating this forest traversal manager.

    Design note: the way trees are serialized is dependent on the
    backing store that is used. For example, the JSON dictionary format is
    for storing trees in Mongo. To decouple this method from the logic
    related to the backing store and minimize information sharing, the
    forest traversal manager would require the get_tree method to
    handle tree de-serialization.

    Design note: the workflow (see README.md) assumes that root nodes
    are added to the forest without using the ForestTraversalManager,
    but with a separate script. Hence, get_forest should be able to
    find trees in this forest. Insted of returning an empty list,
    get_forest should return None if the forest is not found. This
    behavior is intended for datastores that distinguishes between
    file not found versus empty file.

    Design note: the same tree might be uploaded many times for each
    additional node. Because the update of reasoning trees is thread-safe,
    the tree that is uploaded would always be up to date as of the time
    the tree was serialized within the upload_tree method. The update_tree
    method is responsible for handling this concurrency and skipping
    any redundant updates if needed.

    Design note: the consumer of this class is responsible for setting up
    the reasoning_tree_cache, since the consumer might need to set
    cache-specific parameters- especially cache capacity.

    IMPORTANT: get_node goes through tree_cache, and assume that
    pointer objects are returned, so that updating the tree from outside
    would also update the value stored in reasoning tree cache.
    """

    _READ_WRITE_JOB_NAME = "_read_write_job"

    def __init__(
        self,
        reasoning_tree_cache: Cache[ReasoningTreeIdTuple, ReasoningTree],
        get_forest_method: GetForestMethod,
        get_num_children_method: GetNumChildrenMethod,
        upload_tree_method: UploadTreeMethod,
    ):
        self.reasoning_tree_cache = reasoning_tree_cache
        self._get_forest = get_forest_method
        self._get_num_children = get_num_children_method
        self._upload_tree = upload_tree_method

        self.forest_traversal_coordinators: Cache[
            Tuple[ForestName, JobName], ForestTraversalCoordinator
        ] = Cache(capacity=None, get_item_uncached=self._initialize_forest_coordinator)

    def _initialize_forest_coordinator(
        self, forest_job_tuple: Tuple[ForestName, JobName]
    ) -> ForestTraversalCoordinator:
        """
        Initializes forest traversal coordinator for the given
        forest_name.

        TODO: implement asynchronous loading to speed up cold starts.
        (MVP: ai-preference)

        Params:
            forest_job_tuple: ForestName, JobName; JobName is not used.

        Raises:
            KeyError if forest is not found.

        Returns:
            ForestTraversalCoordinator.
        """
        forest_name, _ = forest_job_tuple
        # Bypassing tree cache, retrieve all trees directly using
        # self.get_forest_method.
        trees = self._get_forest(forest_name)
        if trees is None:
            raise KeyError("Forest is not found: {}".format(forest_name))
        elif len(trees) == 0:
            raise ValueError("Forest is empty: {}".format(forest_name))

        # Assemble a list of all nodes in the forest, going through
        # one tree at a time.
        nodes: Dict[NodeTuple, int] = {}
        for tree in trees:
            tree_id: TreeId = tree.attributes["tree_id"]
            assert isinstance(tree_id, int)

            # Ensure all nodes mentioned in edges are
            # actually in nodes.
            for children in tree.edges.values():
                for node_id in children:
                    assert node_id in tree.nodes.keys(), (
                        tree.nodes.keys(),
                        tree.edges.values(),
                    )

            for node_name, node in tree.nodes.items():
                num_children = self._get_num_children(node)
                nodes[(tree_id, node_name)] = num_children

        # Instantiate a forest traversal coordinator for this forest.
        forest_traversal_coordinator = ForestTraversalCoordinator(nodes)
        return forest_traversal_coordinator

    def get_node_read_only(
        self, forest_name: ForestName, job_name: JobName
    ) -> Optional[Tuple[TreeId, Dict, List[PathElement]]]:
        """
        Return tree attribute and a list of nodes,
        representing a path from tree root to the selected node.
        Number of times each path is returned is limited according
        to the max_child specified for the top node of the path
        when that node is added.

        max_children limit is counted separately for each
        (forest_name, job_name) pair.

        IMPORTANT: job_name is used only when forest access is
        read-only.

        The idea is that only one job (the rollout job)
        may add nodes to the forest, whereas there might be more
        than one jobs (AI preference generator) reading from the
        same forest.

        Params:
            forest_name: str

        Returns:
            Tuple:
            - attributes of the tree the top node belongs to
            - list of tree (node_id, node) pairs from root
            to the top node.

            None if no tree node is available.

            Raise KeyError if forest is not found.
        """
        # Retrieve forest traversal coordinator
        # for this (forest_name, job_name) combination.
        traversal_coordinator = self.forest_traversal_coordinators.get(
            (forest_name, job_name)
        )

        # Return None if the forest isn't found.
        if traversal_coordinator is None:
            raise KeyError("No tree is found for forest {}".format(forest_name))

        # Retrieve the next (tree_id, node_identifier)
        # pair from the forest traversal coordinator.
        next_node = traversal_coordinator.get_node()
        if next_node is None:
            return

        tree_id, node_identifier = next_node

        # Retrieve the tree with tree_id from the local tree state cache.
        tree = self.reasoning_tree_cache.get((forest_name, tree_id))
        assert tree is not None

        # Obtain path to the root of the tree.
        # (Implemented within the reasoning tree class.)
        path = tree.get_walk(node_identifier)

        # Return tree attribute and path, without serialization.
        return tree_id, tree.attributes, path

    def get_node_read_write(
        self, forest_name: ForestName
    ) -> Optional[Tuple[TreeId, Dict, List[PathElement]]]:
        """
        Wrapper around get_node_read_only.
        """
        return self.get_node_read_only(forest_name, self._READ_WRITE_JOB_NAME)

    def add_nodes(
        self,
        forest_name: ForestName,
        tree_id: TreeId,
        ancestor_id: NodeId,
        nodes: List[ReasoningTreeNode],
    ) -> List[NodeId]:
        """
        Add node to the given tree in the forest. The number of
        children is obtained from the get_num_children_method
        specified during instantiation.


        For performance reasons, release concurrency lock before
        uploading to DB.

        Params:
            forest_name: Identifier of the forest.
            tree_id: Identifier of the tree.
            ancestor_id: Node identifier of the node that would be the
                the parent of the first new node.
            nodes: List[Node], a list of nodes to add in the same order
                as the list.

        - For each node:
            - Add node to tree.
            - Add node to traversal manager.
        - Upload tree to database using callback method specified when
        instantiating this forest traversal manager.

        Returns:
            List[NodeId], node ids of the selected tree in the same
            order as the given nodes.
        """
        assert len(nodes) > 0, "new node list must be non-empty"

        # Retrieve tree from local cache.
        tree_tuple: ReasoningTreeIdTuple = (forest_name, tree_id)
        tree = self.reasoning_tree_cache.get(tree_tuple)
        if tree is None:
            raise KeyError("Tree not found: {}".format((forest_name, tree_id)))

        # Retrieve the traversal coordinator for
        # tracking edges in the tree.
        traversal_coordinator = self.forest_traversal_coordinators.get(
            (forest_name, self._READ_WRITE_JOB_NAME)
        )
        assert traversal_coordinator is not None

        # Keep track of the node_id of the parent
        # of the next new node.
        parent_node_id = ancestor_id
        new_node_ids: List[NodeId] = []
        for node in nodes:
            # Add node to tree and obtain node_id.
            new_node_id = tree.add_node(node, parent_node_id)
            new_node_ids.append(new_node_id)

            # This new node becomes the parent of the next
            # new node in the list (if applicable.)
            parent_node_id = new_node_id

            # This node is added as part of a walk of nodes.
            # It already has a child. Subtract that child
            # from the max number of children for this node.
            num_additional_children = max(0, self._get_num_children(node) - 1)

            # Add node to traversal coordinator
            new_node_tuple: NodeTuple = (tree_id, new_node_id)
            traversal_coordinator.add_node(new_node_tuple, num_additional_children)

        # Update value only if value is specified in the newly-added leaf.
        if "leaf_value_sum" in nodes[-1].attributes.keys():
            tree.update_branch_values(new_node_ids[-1])

        # Upload tree to database using callback method
        # after adding all new nodes.
        self._upload_tree(tree_tuple, tree)

        return new_node_ids
