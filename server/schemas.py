"""
schemas.py
----------
Pydantic models for the Digital Workspace Architect OpenEnv environment.

All data flowing in/out of the API is validated here. Keeping schemas in a
dedicated module makes it easy to share types between env.py and server.py
without circular imports.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Supported action commands
# ---------------------------------------------------------------------------


class ActionCommand(str, Enum):
    """Every command the agent may issue in a single step."""

    read_file = "read_file"
    move_file = "move_file"
    rename_file = "rename_file"
    create_folder = "create_folder"
    append_to_file = "append_to_file"
    submit_task = "submit_task"


# ---------------------------------------------------------------------------
# Inbound: Agent → Environment
# ---------------------------------------------------------------------------


class WorkspaceAction(BaseModel):
    """
    A single action the agent wants to execute inside the virtual workspace.

    Parameters vary by command:

    | command        | required params                   | optional params |
    |----------------|-----------------------------------|-----------------|
    | read_file      | path                              |                 |
    | move_file      | path, destination                 |                 |
    | rename_file    | path, new_name                    |                 |
    | create_folder  | path                              |                 |
    | append_to_file | path, content                     |                 |
    | submit_task    |                                   |                 |
    """

    command: ActionCommand = Field(..., description="The operation to perform.")

    # Source path (file or folder) – required for most commands.
    path: Optional[str] = Field(
        None,
        description=(
            "Virtual path to the target file/folder, e.g. '/workspace/notes.txt'. "
            "Use forward slashes. Must start with '/workspace'."
        ),
    )

    # Destination folder path used by move_file.
    destination: Optional[str] = Field(
        None,
        description=(
            "Destination *folder* path for move_file, "
            "e.g. '/workspace/Documents'. Must start with '/workspace'."
        ),
    )

    # New bare filename (no path) used by rename_file.
    new_name: Optional[str] = Field(
        None,
        description=(
            "The new bare filename (no directory component) for rename_file, "
            "e.g. '2026-04-Meeting.md'."
        ),
    )

    # Text to append used by append_to_file.
    content: Optional[str] = Field(
        None,
        description="Text content to append to the file (append_to_file only).",
    )

    # ------------------------------------------------------------------ #
    #  Cross-field validation                                             #
    # ------------------------------------------------------------------ #

    @model_validator(mode="after")
    def _check_required_params(self) -> "WorkspaceAction":
        cmd = self.command

        if cmd in (
            ActionCommand.read_file,
            ActionCommand.create_folder,
        ) and not self.path:
            raise ValueError(f"'path' is required for command '{cmd}'.")

        if cmd == ActionCommand.move_file:
            if not self.path:
                raise ValueError("'path' is required for move_file.")
            if not self.destination:
                raise ValueError("'destination' is required for move_file.")

        if cmd == ActionCommand.rename_file:
            if not self.path:
                raise ValueError("'path' is required for rename_file.")
            if not self.new_name:
                raise ValueError("'new_name' is required for rename_file.")

        if cmd == ActionCommand.append_to_file:
            if not self.path:
                raise ValueError("'path' is required for append_to_file.")
            if self.content is None:
                raise ValueError("'content' is required for append_to_file.")

        return self


# ---------------------------------------------------------------------------
# Outbound: Environment → Agent
# ---------------------------------------------------------------------------


class WorkspaceObservation(BaseModel):
    """
    The observation the environment returns after each step.

    The agent receives:
    - The full virtual file-system tree (always present).
    - The content of a file that was just read (only after read_file).
    - A human-readable system message describing what happened.
    - The cumulative reward so far in this episode.
    - Whether the episode has ended (task submitted or hard error).
    - The current task description so the agent always has context.
    """

    # Full directory tree as a nested dict:
    #   {"workspace": {"Documents": {}, "report.md": None}}
    # Folders are dicts; files are None (or their content string for leaf nodes).
    tree: Dict[str, Any] = Field(
        ...,
        description="Nested dict representing the current virtual file system.",
    )

    # Non-None only after a successful read_file command.
    file_content: Optional[str] = Field(
        None,
        description="Contents of the most recently read file, if applicable.",
    )

    # Human-readable feedback (success or error explanation).
    message: str = Field(..., description="Human-readable result of the last action.")

    # Cumulative reward accumulated during this episode.
    reward: float = Field(..., description="Cumulative reward so far in this episode.")

    # True once submit_task is called or the episode is otherwise terminated.
    done: bool = Field(
        False,
        description="True when the episode has finished.",
    )

    # Repeated here so the agent always has it in context.
    task: str = Field(..., description="Current task description.")

    # Step counter.
    step_count: int = Field(..., description="Number of steps taken in this episode.")


# ---------------------------------------------------------------------------
# Request body for /reset
# ---------------------------------------------------------------------------


class ResetRequest(BaseModel):
    """Body sent to POST /reset to start a new episode."""

    task_id: int = Field(
        ...,
        ge=1,
        le=3,
        description="Which task to load (1 = Sorter, 2 = Renamer, 3 = Linker).",
    )


# ---------------------------------------------------------------------------
# Generic error response
# ---------------------------------------------------------------------------


class ErrorResponse(BaseModel):
    """Returned by the API when a request-level error occurs (4xx / 5xx)."""

    detail: str
