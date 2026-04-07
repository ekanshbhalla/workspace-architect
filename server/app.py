"""
server.py
---------
FastAPI application for the Digital Workspace Architect OpenEnv environment.

Endpoints
---------
POST /reset          – Start a new episode (body: {"task_id": 1|2|3})
POST /step           – Execute one action  (body: WorkspaceAction JSON)
GET  /state          – Inspect current state without taking an action
GET  /health         – Lightweight liveness probe
GET  /tasks          – List available tasks and their descriptions

All responses are WorkspaceObservation JSON objects unless otherwise noted.
"""

from __future__ import annotations

from typing import Dict, Any

import uvicorn
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from server.env import WorkspaceEnv
from server.schemas import (
    ErrorResponse,
    ResetRequest,
    WorkspaceAction,
    WorkspaceObservation,
)

# ---------------------------------------------------------------------------
# App & middleware
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Digital Workspace Architect",
    description=(
        "An OpenEnv-compatible environment where an AI agent organises "
        "a virtual file system by reading, moving, renaming, and tagging files."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Shared environment instance
# ---------------------------------------------------------------------------
# A single in-memory environment is shared across all requests.
# For multi-agent / parallel experiments, create one instance per session
# (e.g., keyed by a session_id header) using a dict of envs.

_env = WorkspaceEnv()

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get(
    "/health",
    summary="Liveness probe",
    tags=["Infrastructure"],
)
async def health() -> Dict[str, str]:
    """Returns 200 OK when the server is running."""
    return {"status": "ok"}


@app.get(
    "/tasks",
    summary="List available tasks",
    tags=["Environment"],
)
async def list_tasks() -> Dict[str, Any]:
    """
    Returns descriptions and IDs for all available tasks so agents (and
    developers) can choose the right task_id for /reset.
    """
    from server.env import TASKS  # local import to avoid circular at module level

    return {
        "tasks": [
            {
                "task_id": tid,
                "description": info["description"],
            }
            for tid, info in TASKS.items()
        ]
    }


@app.post(
    "/reset",
    response_model=WorkspaceObservation,
    summary="Reset the environment",
    tags=["Environment"],
    responses={
        400: {"model": ErrorResponse, "description": "Invalid task_id"},
    },
)
async def reset(body: ResetRequest) -> WorkspaceObservation:
    """
    Initialise (or re-initialise) the environment for a given task.

    - Wipes the current virtual file system.
    - Loads the task's initial state.
    - Resets reward and step counter to zero.

    Returns the initial observation (including the task description).
    """
    try:
        obs = _env.reset(task_id=body.task_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    return obs


@app.post(
    "/step",
    response_model=WorkspaceObservation,
    summary="Execute one action",
    tags=["Environment"],
    responses={
        422: {"description": "Validation error – invalid action payload"},
    },
)
async def step(action: WorkspaceAction) -> WorkspaceObservation:
    """
    Execute a single action inside the virtual workspace.

    The environment validates parameters internally and returns:
    - An updated file-system tree.
    - The file content (only after `read_file`).
    - A human-readable feedback message.
    - The cumulative reward and step count.
    - A `done` flag indicating whether the episode has ended.

    If `done` is True, call `/reset` to start a new episode.
    """
    obs = _env.step(action)
    return obs


@app.get(
    "/state",
    response_model=WorkspaceObservation,
    summary="Inspect current state",
    tags=["Environment"],
)
async def state() -> WorkspaceObservation:
    """
    Return the current observation without advancing the step counter
    or modifying the environment.  Useful for debugging or recovering
    context after a disconnection.
    """
    return _env.state()


# ---------------------------------------------------------------------------
# Custom exception handler for unhandled errors
# ---------------------------------------------------------------------------


@app.exception_handler(Exception)
async def generic_exception_handler(request, exc: Exception) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": f"Internal server error: {type(exc).__name__}: {exc}"},
    )


# ---------------------------------------------------------------------------
# Entry point (for local development without Docker)
# ---------------------------------------------------------------------------

def main():
    """Entry point for the OpenEnv validator and server boot."""
    uvicorn.run(
        "server.app:app",
        host="0.0.0.0",
        port=7860,
        reload=False,
        log_level="info",
    )

if __name__ == "__main__":
    main()
