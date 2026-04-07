"""
env.py
------
Core logic for the Digital Workspace Architect OpenEnv environment.

The entire file system lives in a Python dict — no OS calls, no disk I/O.
This guarantees:
  • Deterministic resets.
  • Safety inside a sandboxed Docker container.
  • Fast in-memory operations.

Public interface
----------------
    env = WorkspaceEnv()
    obs  = env.reset(task_id=1)
    obs  = env.step(action)
    obs  = env.state()
"""

from __future__ import annotations

import copy
import re
from typing import Any, Dict, List, Optional, Tuple

from server.schemas import ActionCommand, WorkspaceAction, WorkspaceObservation


# ---------------------------------------------------------------------------
# Reward constants
# ---------------------------------------------------------------------------

REWARD_VALID_ACTION: float = 0.1
REWARD_INVALID_ACTION: float = -0.05
REWARD_TASK_SUCCESS: float = 1.0
REWARD_TASK_FAILURE: float = -0.5

# Maximum steps before an episode is force-terminated.
MAX_STEPS: int = 50

# Root of every virtual path.
WORKSPACE_ROOT: str = "/workspace"


# ---------------------------------------------------------------------------
# Task definitions
# ---------------------------------------------------------------------------

# Each task is a dict with:
#   "description" : str  – shown to the agent
#   "initial_fs"  : dict – starting virtual file system (nested dicts)
#   "file_contents": dict – flat map of virtual_path → file content
#   "grader"      : callable(fs, contents) → (success: bool, detail: str)


def _make_task_1() -> dict:
    """
    Task 1 – The Sorter (Easy)
    --------------------------
    A messy downloads folder containing startup budgets, RC transmitter code, and media.
    Target structure:
        /workspace/
            Code/
                stm32_config.c
                radio_tx.c
            FinTech/
                nadexia_q1_budget.csv
                investment_calc.csv
            Media/
                ypl_podcast_banner.png
    """
    initial_fs: Dict[str, Any] = {
        "stm32_config.c": None,
        "radio_tx.c": None,
        "nadexia_q1_budget.csv": None,
        "investment_calc.csv": None,
        "ypl_podcast_banner.png": None,
    }

    file_contents: Dict[str, str] = {
        f"{WORKSPACE_ROOT}/stm32_config.c": "void SystemClock_Config(void) { // STM32 init }",
        f"{WORKSPACE_ROOT}/radio_tx.c": "#include <nRF24L01.h>\n// Transmitter logic",
        f"{WORKSPACE_ROOT}/nadexia_q1_budget.csv": "expense,amount\nServer Hosting,500\nMarketing,1200",
        f"{WORKSPACE_ROOT}/investment_calc.csv": "month,sip_amount\nJan,2000\nFeb,2000",
        f"{WORKSPACE_ROOT}/ypl_podcast_banner.png": "<binary:png:ypl_banner>",
    }

    target_tree: Dict[str, Any] = {
        "Code": {"stm32_config.c": None, "radio_tx.c": None},
        "FinTech": {"nadexia_q1_budget.csv": None, "investment_calc.csv": None},
        "Media": {"ypl_podcast_banner.png": None},
    }

    def grader(fs: Dict[str, Any], _contents: Dict[str, str]) -> Tuple[float, str]:
        """Calculates a partial score from 0.0 to 1.0 based on correctly moved files."""
        correct_files = 0
        total_files = 5
        
        if "Code" in fs and isinstance(fs["Code"], dict):
            if "stm32_config.c" in fs["Code"]: correct_files += 1
            if "radio_tx.c" in fs["Code"]: correct_files += 1
            
        if "FinTech" in fs and isinstance(fs["FinTech"], dict):
            if "nadexia_q1_budget.csv" in fs["FinTech"]: correct_files += 1
            if "investment_calc.csv" in fs["FinTech"]: correct_files += 1
            
        if "Media" in fs and isinstance(fs["Media"], dict):
            if "ypl_podcast_banner.png" in fs["Media"]: correct_files += 1

        score = correct_files / total_files
        
        if score == 1.0:
            return 1.0, "Perfect organisation! All files are in the correct folders."
        elif score > 0.0:
            return score, f"Partial progress. {correct_files}/{total_files} files in correct folders."
        else:
            return 0.0, "No files are in the correct target folders."

    return {
        "description": (
            "TASK 1 – THE SORTER\n"
            "You have a flat workspace with mixed file types.\n"
            "Your goal:\n"
            "  1. Create three folders: 'Code', 'FinTech', 'Media'.\n"
            "  2. Move .c files → Code/, .csv files → FinTech/, .png files → Media/.\n"
            "Call submit_task when done."
        ),
        "initial_fs": initial_fs,
        "file_contents": file_contents,
        "grader": grader,
    }


def _make_task_2() -> dict:
    """
    Task 2 – The Renamer (Medium)
    ------------------------------
    Lazily-named files.  Agent must read each file, infer the date from the
    content, and rename to YYYY-MM-<Topic>.md format.

    Expected final names:
        2026-01-Meeting.md
        2026-03-Workshop.md
        2026-04-Meeting.md
        2026-02-Review.md
    """
    initial_fs: Dict[str, Any] = {
        "file1.md": None,
        "file2.md": None,
        "file3.md": None,
        "file4.md": None,
    }

    file_contents: Dict[str, str] = {
        f"{WORKSPACE_ROOT}/file1.md": "Meeting notes from January 2026. Topics: roadmap, hiring.",
        f"{WORKSPACE_ROOT}/file2.md": "Workshop recap from March 2026. Topics: design sprint.",
        f"{WORKSPACE_ROOT}/file3.md": "Meeting notes from April 2026. Topics: Q2 planning.",
        f"{WORKSPACE_ROOT}/file4.md": "Quarterly review from February 2026. Topics: OKRs.",
    }

    expected_names = {"2026-01-Meeting.md", "2026-03-Workshop.md",
                      "2026-04-Meeting.md", "2026-02-Review.md"}

    def grader(fs: Dict[str, Any], _contents: Dict[str, str]) -> Tuple[bool, str]:
        actual_names = {k for k, v in fs.items() if v is None}  # files at root
        # Ignore any folders the agent may have created.
        if actual_names == expected_names:
            return True, "All files renamed correctly!"
        missing = expected_names - actual_names
        extra = actual_names - expected_names
        parts: List[str] = []
        if missing:
            parts.append(f"Still missing: {sorted(missing)}")
        if extra:
            parts.append(f"Unexpected files: {sorted(extra)}")
        return False, "Rename incomplete. " + " | ".join(parts)

    return {
        "description": (
            "TASK 2 – THE RENAMER\n"
            "You have four lazily-named Markdown files (file1.md … file4.md).\n"
            "Your goal:\n"
            "  1. Use read_file on each to see its content.\n"
            "  2. Rename each to YYYY-MM-<Topic>.md based on the date and topic in the content.\n"
            "     e.g. 'Meeting notes from April 2026' → '2026-04-Meeting.md'\n"
            "     e.g. 'Workshop recap from March 2026' → '2026-03-Workshop.md'\n"
            "     e.g. 'Quarterly review from February 2026' → '2026-02-Review.md'\n"
            "Call submit_task when all four files are renamed."
        ),
        "initial_fs": initial_fs,
        "file_contents": file_contents,
        "grader": grader,
    }


def _make_task_3() -> dict:
    """
    Task 3 – The Linker (Hard)
    --------------------------
    Daily notes + project notes. Agent must read daily notes, identify mentions
    of the Formula Bharat team, and tag those specific files.
    """
    initial_fs: Dict[str, Any] = {
        "DailyNotes": {
            "2026-03-07.md": None,
            "2026-03-08.md": None,
            "2026-03-09.md": None,
        },
        "Projects": {
            "formula_bharat.md": None,
            "forage_club.md": None,
        },
    }

    file_contents: Dict[str, str] = {
        f"{WORKSPACE_ROOT}/DailyNotes/2026-03-07.md": (
            "Attended the Youth Policy Lab meeting today. Media broadcasting strategy looks good."
        ),
        f"{WORKSPACE_ROOT}/DailyNotes/2026-03-08.md": (
            "Need to draft the budget for the Formula Bharat car design. "
            "Meeting the Dean tomorrow for approval."
        ),
        f"{WORKSPACE_ROOT}/DailyNotes/2026-03-09.md": (
            "Follow up on the Formula Bharat team recruitment. We need more engineers."
        ),
        f"{WORKSPACE_ROOT}/Projects/formula_bharat.md": "# Formula Bharat 2026-27\nGoal: Build a competitive car.",
        f"{WORKSPACE_ROOT}/Projects/forage_club.md": "# Forage Club\nPublic speaking events.",
    }

    _orig_08 = file_contents[f"{WORKSPACE_ROOT}/DailyNotes/2026-03-08.md"]
    _orig_09 = file_contents[f"{WORKSPACE_ROOT}/DailyNotes/2026-03-09.md"]

    def grader(fs: Dict[str, Any], contents: Dict[str, str]) -> Tuple[float, str]:
        tag = "#formula-bharat"
        score = 0.0
        
        p08 = f"{WORKSPACE_ROOT}/DailyNotes/2026-03-08.md"
        p09 = f"{WORKSPACE_ROOT}/DailyNotes/2026-03-09.md"
        
        # Check target files (+0.5 for each correct tag without destroying content)
        for path, orig in [(p08, _orig_08), (p09, _orig_09)]:
            content = contents.get(path, "")
            if tag in content and content.startswith(orig):
                score += 0.5
                
        # Heavy penalty for tagging the wrong file
        if tag in contents.get(f"{WORKSPACE_ROOT}/DailyNotes/2026-03-07.md", ""):
            score -= 0.5
            
        score = max(0.0, min(1.0, score)) # Clamp between 0.0 and 1.0
        
        if score == 1.0: return 1.0, "Correct! Exactly the right files have been tagged."
        return score, f"Partial success. Score: {score}. Check tagging accuracy and data integrity."

    return {
        "description": (
            "TASK 3 – THE LINKER\n"
            "You have a 'DailyNotes' folder (3 entries) and a 'Projects' folder.\n"
            "Your goal:\n"
            "  1. Read every file in DailyNotes/.\n"
            "  2. Identify which notes mention 'Formula Bharat'.\n"
            "  3. Append the tag '#formula-bharat' (on a new line) to ONLY those files.\n"
            "Call submit_task when done."
        ),
        "initial_fs": initial_fs,
        "file_contents": file_contents,
        "grader": grader,
    }


TASKS: Dict[int, dict] = {
    1: _make_task_1(),
    2: _make_task_2(),
    3: _make_task_3(),
}


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _diff_trees(
    expected: Dict[str, Any], actual: Dict[str, Any], prefix: str = ""
) -> Tuple[List[str], List[str]]:
    """
    Recursively compare two FS trees.
    Returns (missing_from_actual, extra_in_actual) path lists.
    """
    missing: List[str] = []
    extra: List[str] = []

    for key, val in expected.items():
        full = f"{prefix}/{key}" if prefix else key
        if key not in actual:
            missing.append(full)
        elif isinstance(val, dict):
            if not isinstance(actual[key], dict):
                missing.append(full)
            else:
                m, e = _diff_trees(val, actual[key], prefix=full)
                missing.extend(m)
                extra.extend(e)

    for key in actual:
        full = f"{prefix}/{key}" if prefix else key
        if key not in expected:
            extra.append(full)

    return missing, extra


def _parse_path(raw: str) -> Optional[List[str]]:
    """
    Convert a virtual path string to a list of path components.

    '/workspace/DailyNotes/2026-04-01.md' → ['DailyNotes', '2026-04-01.md']

    Returns None if the path is invalid or outside /workspace.
    """
    raw = raw.strip()
    if not raw.startswith(WORKSPACE_ROOT):
        return None
    # Strip the root prefix and split.
    relative = raw[len(WORKSPACE_ROOT):]
    parts = [p for p in relative.split("/") if p]
    return parts  # may be empty list (== workspace root itself)


def _get_node(
    fs: Dict[str, Any], parts: List[str]
) -> Tuple[Optional[Any], Optional[Dict[str, Any]], Optional[str]]:
    """
    Walk the FS tree and return (node, parent_dict, node_name).
    Returns (None, None, None) if not found.
    """
    if not parts:
        # Asking for the root itself — not a valid target for file ops.
        return None, None, None

    current: Any = fs
    parent: Optional[Dict[str, Any]] = None
    name: Optional[str] = None

    for part in parts:
        if not isinstance(current, dict) or part not in current:
            return None, None, None
        parent = current
        name = part
        current = current[part]

    return current, parent, name


# ---------------------------------------------------------------------------
# Core environment class
# ---------------------------------------------------------------------------


class WorkspaceEnv:
    """
    In-memory simulation of a file system workspace.

    State is stored in two structures:
      self._fs       – nested dict representing directory tree
                       (folders are dicts, files are None sentinel)
      self._contents – flat dict mapping virtual_path → file content string

    This separation keeps tree operations simple while still allowing
    content reads/writes without encoding content in the tree itself.
    """

    def __init__(self) -> None:
        self._task_id: int = 1
        self._task: dict = TASKS[1]
        self._fs: Dict[str, Any] = {}
        self._contents: Dict[str, str] = {}
        self._reward: float = 0.0
        self._step_count: int = 0
        self._done: bool = False

    # ------------------------------------------------------------------ #
    #  Public interface                                                    #
    # ------------------------------------------------------------------ #

    def reset(self, task_id: int) -> WorkspaceObservation:
        """Start a fresh episode for the given task."""
        if task_id not in TASKS:
            raise ValueError(f"Unknown task_id {task_id}. Must be 1, 2, or 3.")

        self._task_id = task_id
        self._task = TASKS[task_id]
        # Deep-copy so repeated resets are independent.
        self._fs = copy.deepcopy(self._task["initial_fs"])
        self._contents = copy.deepcopy(self._task["file_contents"])
        self._reward = 0.0
        self._step_count = 0
        self._done = False

        return self._build_observation(
            message=f"Environment reset. {self._task['description']}"
        )

    def step(self, action: WorkspaceAction) -> WorkspaceObservation:
        """Execute one action and return the resulting observation."""
        if self._done:
            return self._build_observation(
                message="Episode is already finished. Call /reset to start a new one.",
                file_content=None,
            )

        self._step_count += 1

        # Dispatch to the appropriate handler.
        handler = {
            ActionCommand.read_file: self._handle_read_file,
            ActionCommand.move_file: self._handle_move_file,
            ActionCommand.rename_file: self._handle_rename_file,
            ActionCommand.create_folder: self._handle_create_folder,
            ActionCommand.append_to_file: self._handle_append_to_file,
            ActionCommand.submit_task: self._handle_submit_task,
        }[action.command]

        message, file_content = handler(action)

        # Hard step limit.
        if self._step_count >= MAX_STEPS and not self._done:
            self._done = True
            self._reward += REWARD_TASK_FAILURE
            message += f" [Step limit of {MAX_STEPS} reached — episode terminated.]"

        return self._build_observation(message=message, file_content=file_content)

    def state(self) -> WorkspaceObservation:
        """Return the current observation without advancing the episode."""
        return self._build_observation(message="Current state (no action taken).")

    # ------------------------------------------------------------------ #
    #  Action handlers                                                     #
    # ------------------------------------------------------------------ #

    def _handle_read_file(
        self, action: WorkspaceAction
    ) -> Tuple[str, Optional[str]]:
        parts = _parse_path(action.path)
        if parts is None:
            self._reward += REWARD_INVALID_ACTION
            return (
                f"Invalid path '{action.path}'. Paths must start with '{WORKSPACE_ROOT}'.",
                None,
            )

        node, _parent, _name = _get_node(self._fs, parts)

        if node is None and not isinstance(node, dict):
            # node is None sentinel → it's a file.
            pass  # proceed to content lookup below

        if node is not None and isinstance(node, dict):
            self._reward += REWARD_INVALID_ACTION
            return f"'{action.path}' is a folder, not a file.", None

        if not self._path_exists(parts):
            self._reward += REWARD_INVALID_ACTION
            return f"File not found: '{action.path}'.", None

        content = self._contents.get(action.path)
        if content is None:
            # File exists in tree but no content registered (edge case).
            content = ""

        self._reward += REWARD_VALID_ACTION
        return f"Read '{action.path}' successfully.", content

    def _handle_move_file(
        self, action: WorkspaceAction
    ) -> Tuple[str, Optional[str]]:
        src_parts = _parse_path(action.path)
        dst_parts = _parse_path(action.destination)

        if src_parts is None:
            self._reward += REWARD_INVALID_ACTION
            return f"Invalid source path '{action.path}'.", None
        if dst_parts is None:
            self._reward += REWARD_INVALID_ACTION
            return f"Invalid destination path '{action.destination}'.", None
        if not src_parts:
            self._reward += REWARD_INVALID_ACTION
            return "Cannot move the workspace root.", None

        # Verify source exists and is a file.
        node, parent, name = _get_node(self._fs, src_parts)
        if not self._path_exists(src_parts) or isinstance(node, dict):
            self._reward += REWARD_INVALID_ACTION
            return f"Source file not found or is a folder: '{action.path}'.", None

        # Verify destination exists and is a folder.
        dst_node, _dp, _dn = _get_node(self._fs, dst_parts) if dst_parts else ({}, None, None)
        if dst_parts and (not self._path_exists(dst_parts) or not isinstance(dst_node, dict)):
            self._reward += REWARD_INVALID_ACTION
            return (
                f"Destination folder not found or is a file: '{action.destination}'.",
                None,
            )

        # Perform the move.
        dst_dir: Dict[str, Any] = (
            self._fs if not dst_parts else _get_node(self._fs, dst_parts)[0]
        )

        if name in dst_dir:
            self._reward += REWARD_INVALID_ACTION
            return (
                f"A file named '{name}' already exists in '{action.destination}'.",
                None,
            )

        # Remove from source.
        del parent[name]

        # Add to destination.
        dst_dir[name] = None

        # Update content key.
        old_content_key = action.path
        new_content_key = f"{action.destination}/{name}"
        if old_content_key in self._contents:
            self._contents[new_content_key] = self._contents.pop(old_content_key)

        self._reward += REWARD_VALID_ACTION
        return f"Moved '{action.path}' → '{action.destination}/{name}'.", None

    def _handle_rename_file(
        self, action: WorkspaceAction
    ) -> Tuple[str, Optional[str]]:
        src_parts = _parse_path(action.path)

        if src_parts is None or not src_parts:
            self._reward += REWARD_INVALID_ACTION
            return f"Invalid source path '{action.path}'.", None

        new_name = action.new_name.strip()
        if "/" in new_name or "\\" in new_name:
            self._reward += REWARD_INVALID_ACTION
            return "new_name must be a bare filename with no path separators.", None

        node, parent, old_name = _get_node(self._fs, src_parts)
        if not self._path_exists(src_parts) or isinstance(node, dict):
            self._reward += REWARD_INVALID_ACTION
            return f"Source file not found or is a folder: '{action.path}'.", None

        if new_name in parent:
            self._reward += REWARD_INVALID_ACTION
            return f"A file named '{new_name}' already exists in the same directory.", None

        # Rename in tree.
        parent[new_name] = parent.pop(old_name)

        # Rename in contents map.
        dir_path = action.path[: action.path.rfind("/")]
        old_key = action.path
        new_key = f"{dir_path}/{new_name}"
        if old_key in self._contents:
            self._contents[new_key] = self._contents.pop(old_key)

        self._reward += REWARD_VALID_ACTION
        return f"Renamed '{old_name}' → '{new_name}'.", None

    def _handle_create_folder(
        self, action: WorkspaceAction
    ) -> Tuple[str, Optional[str]]:
        parts = _parse_path(action.path)

        if parts is None or not parts:
            self._reward += REWARD_INVALID_ACTION
            return f"Invalid path '{action.path}'.", None

        # Traverse to parent, creating intermediate dirs is NOT supported
        # (agent must create them one at a time).
        parent_parts = parts[:-1]
        folder_name = parts[-1]

        if parent_parts:
            parent_node, _pp, _pn = _get_node(self._fs, parent_parts)
            if not isinstance(parent_node, dict):
                self._reward += REWARD_INVALID_ACTION
                return (
                    f"Parent directory '{WORKSPACE_ROOT}/{'/'.join(parent_parts)}' "
                    "does not exist.",
                    None,
                )
            parent_dir = parent_node
        else:
            parent_dir = self._fs

        if folder_name in parent_dir:
            self._reward += REWARD_INVALID_ACTION
            return f"'{action.path}' already exists.", None

        parent_dir[folder_name] = {}
        self._reward += REWARD_VALID_ACTION
        return f"Created folder '{action.path}'.", None

    def _handle_append_to_file(
        self, action: WorkspaceAction
    ) -> Tuple[str, Optional[str]]:
        parts = _parse_path(action.path)

        if parts is None or not parts:
            self._reward += REWARD_INVALID_ACTION
            return f"Invalid path '{action.path}'.", None

        node, _parent, _name = _get_node(self._fs, parts)
        if not self._path_exists(parts) or isinstance(node, dict):
            self._reward += REWARD_INVALID_ACTION
            return f"File not found or is a folder: '{action.path}'.", None

        current_content = self._contents.get(action.path, "")
        # Append with a newline separator if content is non-empty.
        separator = "\n" if current_content and not current_content.endswith("\n") else ""
        self._contents[action.path] = current_content + separator + action.content

        self._reward += REWARD_VALID_ACTION
        return f"Appended content to '{action.path}'.", None

    def _handle_submit_task(
        self, action: WorkspaceAction
    ) -> Tuple[str, Optional[str]]:
        grader = self._task["grader"]
        score, detail = grader(self._fs, self._contents)

        if score == 1.0:
            self._reward += REWARD_TASK_SUCCESS
            message = f"✅ Task perfectly complete! {detail} | Total reward: {self._reward:.2f}"
        else:
            # Grant partial reward based on the score
            self._reward += (score * REWARD_TASK_SUCCESS)
            message = f"⚠️ Task finished with partial score ({score}). {detail} | Total reward: {self._reward:.2f}"

        self._done = True
        return message, None

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _path_exists(self, parts: List[str]) -> bool:
        """Return True if the path components resolve to a node in the FS."""
        if not parts:
            return True  # root always exists
        node, _parent, _name = _get_node(self._fs, parts)
        # _get_node returns (None, None, None) if not found; but a file is
        # stored as None-sentinel, so we must check parent to distinguish.
        _, parent, name = _get_node(self._fs, parts)
        return parent is not None and name in parent

    def _build_observation(
        self,
        message: str,
        file_content: Optional[str] = None,
    ) -> WorkspaceObservation:
        return WorkspaceObservation(
            tree=copy.deepcopy(self._fs),
            file_content=file_content,
            message=message,
            reward=round(self._reward, 4),
            done=self._done,
            task=self._task["description"],
            step_count=self._step_count,
        )
