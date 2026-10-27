"""
ToolRegistry — Single Authoritative Tool Registry & Schema Catalog
===================================================================
Provides a unified schema catalog and execution interface for all SPIDY tools.
Implements the Mark XXXIX OR tool dispatch standard:
- Canonical names (e.g. file.search, computer.open_app, system.get_time)
- Synonym/alias resolution (e.g. "search_files" → "file.search")
- JSON Schema definitions for OpenAI / Gemini / Claude function calling
- Strict parameter validation and type coercion
- Real-world execution via SkillRegistry with verification
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import TYPE_CHECKING, Any, Callable

from spidy.logging.logger import get_logger
from spidy.skills.base import SkillContext, SkillResult

if TYPE_CHECKING:
    from spidy.skills.registry import SkillRegistry

log = get_logger(__name__)


@dataclass(frozen=True)
class ToolParam:
    """Specification of a single parameter for a registered tool."""
    name: str
    type_name: str          # "string" | "integer" | "number" | "boolean" | "array" | "object"
    description: str
    required: bool = True
    default: Any = None
    enum: list[str] | None = None


@dataclass(frozen=True)
class ToolDefinition:
    """Specification for a single executable tool in SPIDY."""
    name: str                           # Canonical name, e.g. "file.search"
    description: str                    # LLM-facing description
    skill_action: str                   # Underlying SkillRegistry action, e.g. "search_files"
    parameters: list[ToolParam] = field(default_factory=list)
    permission_tier: str = "T0"         # T0 | T1 | T2 | T3
    aliases: list[str] = field(default_factory=list)
    is_idempotent: bool = False

    def to_openai_schema(self) -> dict[str, Any]:
        """Convert this tool definition to OpenAI/standard function tool schema."""
        properties: dict[str, Any] = {}
        required: list[str] = []

        type_map = {
            "string": "string",
            "path": "string",
            "int": "integer",
            "integer": "integer",
            "float": "number",
            "number": "number",
            "bool": "boolean",
            "boolean": "boolean",
            "list": "array",
            "array": "array",
            "dict": "object",
            "object": "object",
        }

        for param in self.parameters:
            json_type = type_map.get(param.type_name.lower(), "string")
            prop_def: dict[str, Any] = {
                "type": json_type,
                "description": param.description,
            }
            if param.enum:
                prop_def["enum"] = param.enum
            if param.default is not None:
                prop_def["default"] = param.default
            properties[param.name] = prop_def
            if param.required:
                required.append(param.name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }


# ── Canonical Tool Catalog ───────────────────────────────────────────────────

_TOOL_DEFINITIONS: list[ToolDefinition] = [
    # ── File Tools ────────────────────────────────────────────────────────────
    ToolDefinition(
        name="file.search",
        description="Search for files by name, extension, or keywords across Desktop, Documents, Downloads, OneDrive, and workspace.",
        skill_action="search_files",
        permission_tier="T0",
        aliases=["search_files", "find_file", "find_files", "search_file", "find"],
        is_idempotent=True,
        parameters=[
            ToolParam("query", "string", "Search query, keyword, or glob pattern (e.g. 'resume', '*.pdf', 'notes').", required=True),
            ToolParam("path", "string", "Directory path to search. If omitted, searches standard user locations (Desktop, Documents, Downloads).", required=False),
            ToolParam("extensions", "string", "Comma-separated extensions filter (e.g. 'pdf,docx').", required=False),
            ToolParam("max_results", "integer", "Maximum number of results to return (default 10).", required=False, default=10),
        ],
    ),
    ToolDefinition(
        name="file.read",
        description="Read the text content of a file from disk.",
        skill_action="read_file",
        permission_tier="T0",
        aliases=["read_file", "view_file", "cat"],
        is_idempotent=True,
        parameters=[
            ToolParam("path", "string", "Path or filename to read.", required=True),
        ],
    ),
    ToolDefinition(
        name="file.create",
        description="Create a new file or write text content to a file. Verifies disk persistence before completing.",
        skill_action="create_file",
        permission_tier="T1",
        aliases=["create_file", "write_file", "save_file", "make_file"],
        is_idempotent=False,
        parameters=[
            ToolParam("path", "string", "Path or filename of the file to create.", required=True),
            ToolParam("content", "string", "Content to write into the file.", required=False, default=""),
        ],
    ),
    ToolDefinition(
        name="file.open",
        description="Open a file with its default system application.",
        skill_action="open_file",
        permission_tier="T1",
        aliases=["open_file", "launch_file"],
        is_idempotent=True,
        parameters=[
            ToolParam("path", "string", "Path to the file to open.", required=True),
        ],
    ),
    ToolDefinition(
        name="file.create_folder",
        description="Create a new folder on the filesystem (defaults to Desktop if path is relative).",
        skill_action="create_folder",
        permission_tier="T1",
        aliases=["create_folder", "make_folder", "new_folder"],
        is_idempotent=True,
        parameters=[
            ToolParam("folder_name", "string", "Name of the folder to create.", required=True),
            ToolParam("path", "string", "Parent directory. Defaults to Desktop.", required=False),
        ],
    ),
    ToolDefinition(
        name="file.rename_folder",
        description="Rename an existing folder.",
        skill_action="rename_folder",
        permission_tier="T1",
        aliases=["rename_folder"],
        is_idempotent=False,
        parameters=[
            ToolParam("old_name", "string", "Current folder name.", required=True),
            ToolParam("new_name", "string", "New folder name.", required=True),
            ToolParam("path", "string", "Parent directory. Defaults to Desktop.", required=False),
        ],
    ),
    ToolDefinition(
        name="file.delete_folder",
        description="Delete a folder (moves to Recycle Bin safely).",
        skill_action="delete_folder",
        permission_tier="T2",
        aliases=["delete_folder", "remove_folder"],
        is_idempotent=False,
        parameters=[
            ToolParam("folder_name", "string", "Name of the folder to delete.", required=True),
            ToolParam("path", "string", "Parent directory.", required=False),
            ToolParam("force", "boolean", "Permanently delete without Recycle Bin.", required=False, default=False),
        ],
    ),
    ToolDefinition(
        name="file.reveal",
        description="Reveal and select a file in Windows File Explorer.",
        skill_action="reveal_in_explorer",
        permission_tier="T1",
        aliases=["reveal_in_explorer", "show_in_folder", "reveal"],
        is_idempotent=True,
        parameters=[
            ToolParam("path", "string", "Path of the file to reveal in Explorer.", required=True),
        ],
    ),

    # ── Computer & Application Control Tools ──────────────────────────────────
    ToolDefinition(
        name="computer.open_app",
        description="Launch or switch to a desktop application (e.g. 'VS Code', 'Notepad', 'Chrome', 'Edge', 'Calculator').",
        skill_action="launch_app",
        permission_tier="T1",
        aliases=["launch_app", "open_app", "start_app", "app.open"],
        is_idempotent=True,
        parameters=[
            ToolParam("name", "string", "Application name or executable (e.g. 'VS Code', 'Notepad', 'Calculator').", required=True),
            ToolParam("args", "string", "Optional command-line arguments (e.g. file path to open in the app).", required=False, default=""),
        ],
    ),
    ToolDefinition(
        name="computer.close_app",
        description="Close a running application window or process.",
        skill_action="close_app",
        permission_tier="T2",
        aliases=["close_app", "quit_app", "kill_app", "app.close"],
        is_idempotent=True,
        parameters=[
            ToolParam("name", "string", "Application name to close.", required=True),
        ],
    ),
    ToolDefinition(
        name="computer.focus_window",
        description="Bring an application window to the foreground by title or keyword.",
        skill_action="focus_window",
        permission_tier="T1",
        aliases=["focus_window", "bring_to_front"],
        is_idempotent=True,
        parameters=[
            ToolParam("title", "string", "Window title or keyword.", required=True),
        ],
    ),
    ToolDefinition(
        name="computer.minimize_window",
        description="Minimize a window by title or minimize the active window.",
        skill_action="minimize_window",
        permission_tier="T1",
        aliases=["minimize_window"],
        is_idempotent=True,
        parameters=[
            ToolParam("title", "string", "Window title, or empty for the active window.", required=False, default=""),
        ],
    ),
    ToolDefinition(
        name="computer.maximize_window",
        description="Maximize a window by title or maximize the active window.",
        skill_action="maximize_window",
        permission_tier="T1",
        aliases=["maximize_window"],
        is_idempotent=True,
        parameters=[
            ToolParam("title", "string", "Window title, or empty for the active window.", required=False, default=""),
        ],
    ),
    ToolDefinition(
        name="computer.type",
        description="Type text into the currently active focused application window.",
        skill_action="keyboard_type",
        permission_tier="T1",
        aliases=["keyboard_type", "type_text", "type"],
        is_idempotent=False,
        parameters=[
            ToolParam("text", "string", "Text to type into the focused window.", required=True),
            ToolParam("press_enter", "boolean", "Whether to press Enter after typing.", required=False, default=False),
        ],
    ),
    ToolDefinition(
        name="computer.press_key",
        description="Press a single key (e.g. 'enter', 'esc', 'tab', 'backspace', 'f5').",
        skill_action="key_press",
        permission_tier="T1",
        aliases=["key_press", "press_key"],
        is_idempotent=False,
        parameters=[
            ToolParam("key", "string", "Key name to press.", required=True),
        ],
    ),
    ToolDefinition(
        name="computer.hotkey",
        description="Press a keyboard shortcut combination (e.g. 'ctrl+c', 'alt+tab', 'win+r', 'ctrl+v').",
        skill_action="hotkey",
        permission_tier="T1",
        aliases=["hotkey", "press_hotkey"],
        is_idempotent=False,
        parameters=[
            ToolParam("keys", "string", "Keys separated by '+' or spaces (e.g. 'ctrl+s').", required=True),
        ],
    ),
    ToolDefinition(
        name="computer.click",
        description="Click the mouse at specified screen coordinates or the current position.",
        skill_action="mouse_click",
        permission_tier="T1",
        aliases=["mouse_click", "click"],
        is_idempotent=False,
        parameters=[
            ToolParam("x", "integer", "X screen coordinate.", required=False),
            ToolParam("y", "integer", "Y screen coordinate.", required=False),
            ToolParam("button", "string", "Mouse button: 'left' | 'right' | 'middle'.", required=False, default="left"),
            ToolParam("clicks", "integer", "Number of clicks (1 for single, 2 for double).", required=False, default=1),
        ],
    ),

    # ── Browser Tools ─────────────────────────────────────────────────────────
    ToolDefinition(
        name="browser.open",
        description="Open a URL in the browser (Chrome, Edge, etc.).",
        skill_action="open_url",
        permission_tier="T1",
        aliases=["open_url", "navigate", "browse"],
        is_idempotent=True,
        parameters=[
            ToolParam("url", "string", "The URL or website domain to open (e.g. 'https://github.com').", required=True),
            ToolParam("browser_type", "string", "Optional browser type ('chrome', 'edge', 'chromium').", required=False, default="chrome"),
        ],
    ),
    ToolDefinition(
        name="browser.search",
        description="Search Google using the default browser.",
        skill_action="search_google",
        permission_tier="T0",
        aliases=["search_google", "search_web", "web_search", "google"],
        is_idempotent=True,
        parameters=[
            ToolParam("query", "string", "Search query or topic.", required=True),
        ],
    ),
    ToolDefinition(
        name="browser.search_youtube",
        description="Search YouTube and open results in the browser.",
        skill_action="search_youtube",
        permission_tier="T0",
        aliases=["search_youtube", "youtube"],
        is_idempotent=True,
        parameters=[
            ToolParam("query", "string", "Video or channel search query.", required=True),
        ],
    ),

    # ── System & Utility Tools ────────────────────────────────────────────────
    ToolDefinition(
        name="system.get_time",
        description="Get the current local time in the user's timezone.",
        skill_action="get_time",
        permission_tier="T0",
        aliases=["get_time", "time"],
        is_idempotent=True,
        parameters=[],
    ),
    ToolDefinition(
        name="system.get_date",
        description="Get the current date and day of the week in the user's timezone.",
        skill_action="get_date",
        permission_tier="T0",
        aliases=["get_date", "date"],
        is_idempotent=True,
        parameters=[],
    ),
    ToolDefinition(
        name="system.get_info",
        description="Get current system metrics (CPU usage, memory, OS version, battery).",
        skill_action="get_system_info",
        permission_tier="T0",
        aliases=["get_system_info", "system_info"],
        is_idempotent=True,
        parameters=[],
    ),
    ToolDefinition(
        name="calculator.calculate",
        description="Evaluate a mathematical expression deterministically.",
        skill_action="calculate",
        permission_tier="T0",
        aliases=["calculate", "math"],
        is_idempotent=True,
        parameters=[
            ToolParam("expression", "string", "Mathematical expression to evaluate (e.g. '125 * 24').", required=True),
        ],
    ),
    ToolDefinition(
        name="note.create",
        description="Save a quick text note to disk.",
        skill_action="create_note",
        permission_tier="T1",
        aliases=["create_note", "take_note"],
        is_idempotent=False,
        parameters=[
            ToolParam("content", "string", "Note content to save.", required=True),
            ToolParam("title", "string", "Optional title for the note.", required=False, default=""),
        ],
    ),
    ToolDefinition(
        name="screenshot.capture",
        description="Capture a full-screen screenshot and save to disk.",
        skill_action="capture_screen",
        permission_tier="T0",
        aliases=["capture_screen", "take_screenshot", "screenshot"],
        is_idempotent=False,
        parameters=[
            ToolParam("path", "string", "Optional file path to save the screenshot.", required=False, default=""),
        ],
    ),

    # ── Shell Execution Tool (Controlled Fallback with Policy) ─────────────────
    ToolDefinition(
        name="shell.execute",
        description=(
            "Execute a shell command. Use this ONLY as a fallback when no dedicated tool "
            "(such as file.create, file.read, file.search, computer.open_app) exists. "
            "Never use shell to create simple files or search standard locations."
        ),
        skill_action="execute_shell_command",
        permission_tier="T2",
        aliases=["execute_command", "run_command", "run_shell_command", "shell"],
        is_idempotent=False,
        parameters=[
            ToolParam("command", "string", "Command line string to execute.", required=True),
            ToolParam("cwd", "string", "Working directory.", required=False, default=""),
            ToolParam("timeout_seconds", "integer", "Command timeout in seconds (default 30).", required=False, default=30),
        ],
    ),
]


class ToolRegistry:
    """
    Authoritative catalog of tools and schema registry for SPIDY.
    """

    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}
        self._alias_map: dict[str, str] = {}

        for tool in _TOOL_DEFINITIONS:
            self.register(tool)

    def register(self, tool: ToolDefinition) -> None:
        """Register a tool definition and index its aliases."""
        self._tools[tool.name.lower()] = tool
        self._alias_map[tool.name.lower()] = tool.name.lower()
        self._alias_map[tool.skill_action.lower()] = tool.name.lower()
        for alias in tool.aliases:
            self._alias_map[alias.lower()] = tool.name.lower()

    def get_tool(self, name_or_alias: str) -> ToolDefinition | None:
        """Find a ToolDefinition by canonical name, skill action, or alias."""
        clean = (name_or_alias or "").strip().lower()
        canonical_name = self._alias_map.get(clean)
        if canonical_name:
            return self._tools.get(canonical_name)
        return self._tools.get(clean)

    def get_all_tools(self) -> list[ToolDefinition]:
        """Return all registered tool definitions."""
        return list(self._tools.values())

    def get_openai_tool_schemas(self) -> list[dict[str, Any]]:
        """Return all tool schemas in OpenAI / function-calling format."""
        return [tool.to_openai_schema() for tool in self._tools.values()]

    def is_known_tool(self, name_or_alias: str) -> bool:
        """Check whether a tool name or alias is registered."""
        return self.get_tool(name_or_alias) is not None

    def validate_parameters(self, tool_def: ToolDefinition, args: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
        """
        Validate and coerce arguments for a tool call against its definition.
        Returns (coerced_args, error_messages).
        """
        coerced: dict[str, Any] = dict(args)
        errors: list[str] = []

        for param in tool_def.parameters:
            val = coerced.get(param.name)
            if val is None:
                if param.required:
                    errors.append(f"Missing required parameter '{param.name}' for tool '{tool_def.name}'.")
                elif param.default is not None:
                    coerced[param.name] = param.default
                continue

            # Type coercion
            t = param.type_name.lower()
            try:
                if t in ("int", "integer"):
                    coerced[param.name] = int(val)
                elif t in ("float", "number"):
                    coerced[param.name] = float(val)
                elif t in ("bool", "boolean"):
                    if isinstance(val, str):
                        coerced[param.name] = val.lower() in ("true", "1", "yes")
                    else:
                        coerced[param.name] = bool(val)
                elif t in ("str", "string", "path"):
                    coerced[param.name] = str(val).strip()
            except (ValueError, TypeError) as exc:
                errors.append(f"Parameter '{param.name}' expected {t}, got invalid value {val!r} ({exc})")

        return coerced, errors


# Global singleton tool registry
GLOBAL_TOOL_REGISTRY = ToolRegistry()
