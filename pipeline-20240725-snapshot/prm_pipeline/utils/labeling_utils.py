from typing import Any, Dict, List, NamedTuple, Optional, TypeVar

from pymongo.collection import Collection

from ..utils.tree_utils import ForestName, NodeId, TreeId


class ManualLabel(NamedTuple):
    """
    Manual label for a particular example.

    node_id: refers to the node at the top of the path.
    """

    forest_name: ForestName
    tree_id: TreeId
    node_id: NodeId

    config_version: str
    config_commit_hash: str
    annotator: str
    attribute_name: str
    attribute_value: Optional[str]
    data_source_name: Optional[str] = None
    comment: Optional[str] = None

    def get_query_filter(self) -> Dict[str, Any]:
        """
        Return MongoDB filter for selecting self.
        """
        output = self._asdict()
        output.pop("attribute_value")
        output.pop("data_source_name")
        output.pop("comment")
        output.pop("config_commit_hash")

        return output

    def get_current_value(self, collection: Collection) -> Optional["ManualLabel"]:
        """
        Try retrieving an instance matching self.get_filter
        from the given DB collection. If matched, return
        attribute_value of the match. Otherwise, return None.

        Params:
            collection: MongoDB collection.

        Returns:
            matched ManualLabel.
        """
        query = self.get_query_filter()
        match_serialized = collection.find_one(query)

        if match_serialized is None:
            return None

        match_serialized.pop("_id")
        match = ManualLabel(**match_serialized)
        return match

    def upsert(self, collection: Collection):
        """
        Upsert self to collection,
        overwritting existing entry if matched.

        Params:
            collection: MongoDB collection.
        """
        collection.find_one_and_replace(
            self.get_query_filter(),
            self._asdict(),
            upsert=True,
        )
        print("Uploaded:", self._asdict())
