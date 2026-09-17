"""
Jinja2-based template engine for code generation.

This module provides a clean abstraction for generating deployable trading code
using Jinja2 templates, replacing the string.Template approach.
"""

import datetime
import uuid
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, Template, select_autoescape


def python_to_csharp_value(value: Any) -> str:
    """
    Convert Python values to C# syntax.

    Args:
        value: Python value to convert

    Returns:
        C# representation as a string
    """
    if value is None:
        return "null"
    elif isinstance(value, bool):
        return "true" if value else "false"
    elif isinstance(value, str):
        return value
    else:
        return str(value)


class TemplateEngine:
    """
    Jinja2-based template engine for generating trading code.

    This class provides a flexible and maintainable way to generate code
    for different trading platforms using Jinja2 templates.
    """

    def __init__(self, template_dir: str | None = None):
        """
        Initialize the template engine.

        Args:
            template_dir: Directory containing Jinja2 templates. If None, uses
                         the default templates in the translator_engine directory.
        """
        if template_dir is None:
            # Use the directory where this module is located
            template_dir = str(Path(__file__).parent)

        self.template_dir = template_dir

        # Create custom environment with filters
        self.env = Environment(
            loader=FileSystemLoader(template_dir),
            autoescape=select_autoescape(["html", "xml"]),
            trim_blocks=True,
            lstrip_blocks=True,
        )

        # Register custom filters
        self.env.filters["csharp"] = python_to_csharp_value
        self.env.filters["indent"] = self._indent_filter

    def _indent_filter(self, text: str, width: int = 4) -> str:
        """
        Custom indent filter to add indentation to each line.

        Args:
            text: Text to indent
            width: Number of spaces for indentation

        Returns:
            Indented text with each line prefixed by the specified number of spaces
        """
        indent_str = " " * width
        lines = text.split("\n")
        # Indent all lines including the first one
        return "\n".join(indent_str + line for line in lines)

    def render_template(self, template_name: str, **context: Any) -> str:
        """
        Render a template with the given context.

        Args:
            template_name: Name of the template file (e.g., 'ctrader_template.j2')
            **context: Keyword arguments to pass to the template

        Returns:
            Rendered template as a string
        """
        template = self.env.get_template(template_name)
        return template.render(**context)

    def render_string(self, template_string: str, **context: Any) -> str:
        """
        Render a template string with the given context.

        Args:
            template_string: Jinja2 template as a string
            **context: Keyword arguments to pass to the template

        Returns:
            Rendered template as a string
        """
        template = Template(template_string)
        return template.render(**context)

    def save_rendered_code(
        self, rendered_code: str, directory: str, prefix: str = "strategy", extension: str = "txt"
    ) -> str:
        """
        Save rendered code to a file with a unique filename.

        Args:
            rendered_code: The rendered code to save
            directory: Directory to save the file
            prefix: Prefix for the filename
            extension: File extension

        Returns:
            Path to the saved file
        """
        # Create the directory if it doesn't exist
        output_dir = Path(directory)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate a unique filename based on timestamp and UUID
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        filename = f"{prefix}_{timestamp}_{unique_id}.{extension}"
        filepath = output_dir / filename

        # Save the rendered code
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(rendered_code)

        return str(filepath)


class CTraderJinjaTranslator:
    """
    cTrader code generator using Jinja2 templates.

    This class translates evolved trading strategies into cTrader C# code
    using Jinja2 templates for better maintainability and flexibility.
    """

    DEFAULT_STRATEGY_PARAMS = {
        "strategy_name": "EvoStrategy",
        "direction": "LongOnly",
        "trade_size": 0.99,
        "exit_encoded_entry": False,
        "exit_after_n_bars": None,
        "exit_after_n_days": None,
        "exit_end_of_week": False,
        "exit_end_of_month": False,
        "exit_when_pnl_lessthan": None,
        "stop_loss": None,
        "take_profit": None,
        "enable_tsl": False,
    }

    def __init__(self, indicator_signal_code: str, strategy_params: dict | None = None):
        """
        Initialize the cTrader translator.

        Args:
            indicator_signal_code: Generated indicator and signal code
            strategy_params: Strategy configuration parameters
        """
        self.indicator_signal_code = indicator_signal_code
        # Apply C# value conversion to all strategy params
        raw_params = {**self.DEFAULT_STRATEGY_PARAMS, **(strategy_params or {})}
        self.strategy_params = {
            key: python_to_csharp_value(value) if not isinstance(value, str) else value
            for key, value in raw_params.items()
        }
        self.template_engine = TemplateEngine()

    def generate_code(self, root_signal: str, save_directory: str | None = None) -> str:
        """
        Generate complete cTrader C# code.

        Args:
            root_signal: The root signal expression
            save_directory: Optional directory to save the generated code

        Returns:
            Generated cTrader C# code
        """
        # Prepare template context
        context = {
            "indicator_signal_code": self.indicator_signal_code,
            "root_signal": root_signal,
            **self.strategy_params,
        }

        # Render the template
        rendered_code = self.template_engine.render_template("ctrader_template.j2", **context)

        # Save if directory is provided
        if save_directory:
            self.template_engine.save_rendered_code(
                rendered_code, save_directory, prefix="ctrader_code", extension="cs"
            )

        return rendered_code
