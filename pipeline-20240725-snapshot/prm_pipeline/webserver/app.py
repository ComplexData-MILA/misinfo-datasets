import datetime
import os
from typing import Tuple

from flask import Flask

from ..utils import ForestTraversalManager
from .db_utils import MongoReasoningTreeDB, mongo
from .views import bp

# TODO: implement data validation


def get_db_parameters(testing: bool) -> Tuple[str, str]:
    """
    Retrieve DB parameters from environment variables.

    Returns:
        mongo_server, db_name
    """
    mongo_server = os.environ["MONGO_DB_SERVER"]
    if testing:
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        if "TEST_MONGO_DB_NAME" in os.environ.keys():
            os.environ["EXTERNAL_TEST_DB"] = "1"

        db_name = os.environ.get("TEST_MONGO_DB_NAME", "_test_{}".format(timestamp))
    else:
        db_name = os.environ.get("MONGO_DB_NAME", "production")

    return mongo_server, db_name


def create_app(debug: bool = False, testing: bool = False):
    """
    Debug: enable Flask debugger
    Testing: use temporary database collections,
        for unit testing.
    """

    app = Flask(__name__)

    if debug:
        app.config["DEBUG"] = True

    mongo_server, db_name = get_db_parameters(testing)
    app.config["MONGO_URI"] = mongo_server + db_name + "?authSource=admin"
    app.config["MONGO_DB_NAME"] = db_name
    mongo.init_app(app)
    assert mongo.db is not None

    # Instantiate MongoReasoningTreeDB and share this instance
    # through app.config.
    cache_capacity = int(os.environ.get("TREE_CACHE_CAPACITY", 8192))
    reasoning_tree_db = MongoReasoningTreeDB(mongo.db, cache_capacity)
    app.config["REASONING_TREE_DB"] = reasoning_tree_db

    app.register_blueprint(bp)
    return app
