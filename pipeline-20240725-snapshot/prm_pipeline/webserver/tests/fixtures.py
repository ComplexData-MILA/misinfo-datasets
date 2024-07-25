import logging
import os

import pytest

from prm_pipeline.webserver.app import create_app
from prm_pipeline.webserver.db_utils import mongo

logger = logging.getLogger(__name__)


@pytest.fixture()
def app():
    app = create_app(testing=True)
    mongo_db_name: str = app.config["MONGO_DB_NAME"]

    if (
        not mongo_db_name.startswith("_test_")
        or "EXTERNAL_TEST_DB" in os.environ.keys()
    ):
        raise AssertionError(
            "MONGO_DB_NAME {} is external. Please set TEST_KEEP_DB=1.".format(
                mongo_db_name
            )
        )

    # Populate database
    # Create example rollouts
    assert mongo.db is not None
    yield app

    # Clean up database
    if not os.environ.get("TEST_KEEP_DB") == "1":
        mongo.db.command("dropDatabase")
    else:
        logger.warning("Env var TEST_KEEP_DB=1 is set; will not clean up database.")
        logger.warning("Database name: {}".format(app.config["MONGO_DB_NAME"]))
