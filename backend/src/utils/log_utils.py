import logging
import uuid
from datetime import datetime, timezone
from src.utils.connection_utils import db

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

logger = logging.getLogger(__name__)


def insert_logs(message, logType=None, feature=None, bulkId=None, fileId=None, userId=None):
    log_type = (logType or "information").lower()

    # Console output
    if log_type == "error":
        logger.error(message)
    elif log_type == "critical":
        logger.critical(message)
    elif log_type == "warning":
        logger.warning(message)
    else:
        logger.info(message)

    # Persist errors and criticals to MongoDB for post-mortem debugging
    if log_type in ("error", "critical"):
        try:
            db["auditlogs"].insert_one({
                "auditLogId": str(uuid.uuid4()),
                "message": message,
                "logType": log_type,
                "feature": feature or "unspecified",
                "bulkId": str(bulkId) if bulkId else None,
                "fileId": str(fileId) if fileId else None,
                "userId": str(userId) if userId else None,
                "timestamp": datetime.now(timezone.utc),
            })
        except Exception:
            pass  # never let logging crash the app
