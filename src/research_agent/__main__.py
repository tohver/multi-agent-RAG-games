"""Lets the package be run directly: `python -m research_agent`.

The installed `research-agent` command and this entry point both end up in
`cli.main`, so they behave identically.
"""

from .cli import main

raise SystemExit(main())
