"""The shared Rich console and the panels every command prints.

Lives outside cli.py so the pipeline and agents can print without importing the CLI.
"""

from datetime import datetime, timezone

from rich import box
from rich.console import Console
from rich.panel import Panel

console = Console()


def get_bias_color(bias: str) -> str:
    """Panel colour for an agent bias."""
    if bias in ("STRONGLY_BULLISH", "BULLISH"):
        return "green"
    elif bias in ("STRONGLY_BEARISH", "BEARISH"):
        return "red"
    return "yellow"


def format_pipe_string(text: str) -> str:
    """
    Splits dense, pipe-separated string outputs into a clean, multi-line format
    with bullet points for terminal UI readability.
    """
    if not text:
        return ""
    # Split by pipe and clean up whitespace
    segments = [seg.strip() for seg in text.split('|') if seg.strip()]
    # Rejoin with newlines and bullet points
    return "\n".join(f"  • {seg}" for seg in segments)


def render_execution_ticket(ticket: dict, strategy: str) -> None:
    """
    Centralized function to render an execution ticket cleanly.
    """
    verdict = ticket.get("verdict", "")
    panel_color = "green" if verdict == "GO LONG" else "red" if verdict == "GO SHORT" else "yellow"

    timestamp = ticket.get("entry_timestamp", datetime.now().timestamp())
    opened_str = datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')

    operator_summary_lines = [
        f"[bold {panel_color}]Verdict:[/] {verdict}\n",
        f"[bold {panel_color}]Opened:[/] {opened_str}"
    ]

    close_timestamp = ticket.get("exit_timestamp") or ticket.get("close_timestamp")
    if close_timestamp:
        # close_timestamp is stored in seconds (candle ms / 1000 in the resolution check).
        # If it happens to be > 1e11 (unlikely to be valid seconds unless far future, likely ms), handle it gracefully
        if close_timestamp > 1e11:
            close_timestamp /= 1000.0
        closed_str = datetime.fromtimestamp(close_timestamp, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
        operator_summary_lines.append(f"[bold {panel_color}]Closed:[/] {closed_str}")

    operator_summary_lines.extend([
        f"[bold {panel_color}]Order Type:[/] {ticket.get('order_type', '')}",
        f"[bold {panel_color}]Entry Price:[/] ${ticket.get('entry_price', 0):,.2f}",
        f"[bold {panel_color}]Stop Loss:[/] ${ticket.get('stop_loss', 0):,.2f}",
        f"[bold {panel_color}]Take Profit:[/] ${ticket.get('take_profit', 0):,.2f}",
        f"[bold {panel_color}]Risk/Reward Ratio:[/] {ticket.get('risk_reward_ratio', 0):.2f}",
        f"[bold {panel_color}]Position Size USD:[/] ${ticket.get('position_size_usd', 0):,.2f}"
    ])

    operator_summary = "\n".join(operator_summary_lines)

    # The panel draws its own rounded border, so the title only needs plain text.
    # (Adding our own corner characters here made two borders overlap.)
    panel_title = f"[ {strategy.upper()} Operator: Execution Ticket ]"
    console.print(Panel(operator_summary, title=panel_title, border_style=panel_color, box=box.ROUNDED, expand=False))


def render_technical_report(tech_report: dict) -> None:
    """Print the Agent 1 (technical) panel."""
    tech_bias = tech_report.get('bias', 'NEUTRAL')
    tech_color = get_bias_color(tech_bias)

    tech_summary = (
        f"[bold]General Analysis:[/bold]\n{tech_report.get('general_analysis', '')}\n\n"
        f"Bias: [{tech_color} bold]{tech_bias}[/{tech_color} bold]\n\n"
        f"[bold]Trend State:[/bold]\n{format_pipe_string(tech_report.get('trend_state', ''))}\n\n"
        f"[bold]Momentum Divergence:[/bold]\n{format_pipe_string(tech_report.get('momentum_divergence', ''))}\n\n"
        f"[bold]Key Level Interaction:[/bold]\n{format_pipe_string(tech_report.get('key_level_interaction', ''))}"
    )

    console.print(Panel(tech_summary, title="[Technical Analysis Agent]", border_style=tech_color, box=box.ROUNDED, expand=False))


def render_volume_report(vol_report: dict) -> None:
    """Print the Agent 2 (volume and liquidity) panel."""
    vol_bias = vol_report.get('bias', 'NEUTRAL')
    vol_color = get_bias_color(vol_bias)

    vol_summary = (
        f"[bold]General Analysis:[/bold]\n{vol_report.get('general_analysis', '')}\n\n"
        f"Bias: [{vol_color} bold]{vol_bias}[/{vol_color} bold]\n\n"
        f"[bold]Liquidity State:[/bold]\n{format_pipe_string(vol_report.get('liquidity_state', ''))}\n\n"
        f"[bold]Volume Momentum:[/bold]\n{format_pipe_string(vol_report.get('volume_momentum', ''))}\n\n"
        f"[bold]Magnet Target:[/bold]\n{format_pipe_string(vol_report.get('magnet_target', ''))}"
    )

    console.print(Panel(vol_summary, title="[Volume & Liquidity Agent]", border_style=vol_color, box=box.ROUNDED, expand=False))


def render_manager_report(report: dict, title: str, border_style: str) -> None:
    """Print an Agent 3 (defensive or greedy manager) panel."""
    verdict = report.get('final_verdict', 'SIT ON HANDS')
    summary = (
        f"[bold]Executive Summary:[/bold]\n{report.get('executive_summary', '')}\n\n"
        f"[bold]Confluence Matrix:[/bold]\n{format_pipe_string(report.get('confluence_matrix', ''))}\n\n"
        f"[bold]Risk Vector:[/bold]\n{format_pipe_string(report.get('risk_vector', ''))}\n\n"
        f"[bold]Final Verdict:[/bold] {verdict}"
    )

    console.print(Panel(summary, title=title, border_style=border_style, box=box.ROUNDED, expand=False))


def render_invalid_ticket(verdict: str, current_price, threat_level, magnet_target, stop_loss: float, take_profit: float) -> None:
    """Print the operator's diagnostic panel for a ticket that contradicts its direction."""
    diagnostic_text = (
        f"[bold red]⛔ INVALID TICKET LOGIC DETECTED[/bold red]\n\n"
        f"Verdict: {verdict}\n"
        f"Current Price: {current_price}\n"
        f"Threat Level (from Agent 1): {threat_level} (Type: {type(threat_level)})\n"
        f"Magnet Target (from Agent 2): {magnet_target} (Type: {type(magnet_target)})\n"
        f"Calculated Stop Loss: {stop_loss}\n"
        f"Calculated Take Profit: {take_profit}"
    )
    console.print(Panel(diagnostic_text, border_style="red", box=box.ROUNDED, expand=False))
