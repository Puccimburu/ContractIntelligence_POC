from fastapi import APIRouter

router = APIRouter()

@router.get("/", summary="Health check")
async def root():
    return {
        "service": "Contract Intelligence — Legal Desk API",
        "status": "running",
        "endpoints": {
            "upload": "POST /ci/upload",
            "status": "GET  /ci/status/{fileId}",
            "file":   "GET  /ci/file/{fileId}",
            "ask":    "POST /desk/conversation",
            "stream": "GET  /desk/messages/stream/{aiMessageId}",
        },
    }
