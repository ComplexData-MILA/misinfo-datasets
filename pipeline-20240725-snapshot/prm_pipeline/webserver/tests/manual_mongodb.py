import logging
import os

import pymongo

MONGO_DB_URL = os.environ.get("MONGO_DB_URL")
assert MONGO_DB_URL is not None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info("MONGO_DB_URL: {}".format(MONGO_DB_URL))

client = pymongo.MongoClient(MONGO_DB_URL)
