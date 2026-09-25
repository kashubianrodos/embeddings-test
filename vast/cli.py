#!/usr/bin/env python3
"""vast.ai LLM benchmark — command-line front end.

    ./vastbench models                      # what can be benchmarked
    ./vastbench bench                       # pick model → cards → disk interactively (Enter = default)
    ./vastbench bench bielik-1.5b-q8 --yes  # no questions, all defaults
    ./vastbench offers qwen3.8-27b-q4k --gpu 4090     # ranked offers, nothing rented
    ./vastbench bench bielik-11b --simulate gx10      # + projection onto ASUS Ascent GX10 (GB10)
    ./vastbench project 52613296                      # project an existing run onto the GX10

The heavy lifting (offer ranking, renting, polling, log fetching, guaranteed destroy) stays in
the tested bash scripts in vast/. This CLI chooses what to run and hands the choice over as
RUN_* environment variables, which win over presets and vast/.env.
"""
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.prompt import IntPrompt, Prompt
from rich.table import Table

VAST_DIR = Path(__file__).resolve().parent
ROOT = VAST_DIR.parent
sys.path.insert(0, str(VAST_DIR))
from models import (DEFAULT_MODEL, GPUS, MODELS, Model, find_model,  # noqa: E402
                    normalize_gpu)
from project import TARGETS, project_run, write_report  # noqa: E402

app = typer.Typer(add_completion=False, no_args_is_help=True, rich_markup_mode="rich",
                  help="Benchmark LLMs on rented vast.ai GPUs — cheapest run, always destroyed at the end.")
console = Console(stderr=True)


# ----------------------------------------------------------------------------- helpers
def interactive(no_input: bool) -> bool:
    return not no_input and sys.stdin.isatty() and sys.stdout.isatty()


def model_table() -> Table:
    t = Table(title="Models", title_justify="left")
    for col in ("#", "model", "engine", "weights", "min VRAM", "default cards", "disk", "note"):
        t.add_column(col, justify="right" if col in ("#", "weights", "min VRAM", "disk") else "left")
    for i, m in enumerate(MODELS, 1):
        t.add_row(str(i), m.key, m.engine, f"{m.weights_gb:g} GB", f"{m.min_vram_gb} GB",
                  ", ".join(m.default_gpus) or f"any ≥{m.min_vram_gb} GB", f"{m.disk_gb} GB", m.note)
    return t


def gpu_table(m: Model, simulate: Optional[str] = None) -> Table:
    t = Table(title=f"Cards for {m.key} (≥{m.min_vram_gb} GB"
                    + (f", compute ≥{m.min_compute / 100:g}" if m.min_compute else "") + ")",
              title_justify="left")
    for col in ("#", "card", "VRAM", "bandwidth", "compute"):
        t.add_column(col, justify="left" if col == "card" else "right")
    t.add_row("0", "[bold]any compatible[/] — the ranker picks the cheapest run", "", "", "")
    tbw = TARGETS[simulate].bw_spec_gbs if simulate else None
    for i, g in enumerate(m.compatible_gpus(), 1):
        mark = "  [cyan]≈ " + simulate.upper() + " bandwidth[/]" if tbw and g.bw_gbs and g.bw_gbs <= tbw * 1.7 else ""
        t.add_row(str(i), g.name + mark, f"{g.vram_gb} GB", f"{g.bw_gbs:g} GB/s" if g.bw_gbs else "?",
                  f"{g.compute_cap / 100:g}")
    return t


def resolve_model(name: Optional[str], ask: bool) -> Model:
    if name:
        m = find_model(name)
        if not m:
            raise typer.BadParameter(f"unknown model '{name}'. Choose from: {', '.join(x.key for x in MODELS)}")
        return m
    if not ask:
        return find_model(DEFAULT_MODEL)
    console.print(model_table())
    while True:
        m = find_model(Prompt.ask("Model (number or name)", default="1", console=console))
        if m:
            return m
        console.print("[red]Not in the list.[/]")


def parse_gpus(m: Model, values: List[str]) -> List[str]:
    """Accept names ('4090', 'RTX_4090'), list numbers from gpu_table, or 'any'/'0'."""
    compat = m.compatible_gpus()
    out: List[str] = []
    for v in values:
        for part in str(v).replace(";", ",").split(","):
            part = part.strip()
            if not part:
                continue
            if part.lower() in ("0", "any", "all"):
                return []
            if part.isdigit() and 1 <= int(part) <= len(compat):
                out.append(compat[int(part) - 1].name)
                continue
            name = normalize_gpu(part)
            g = GPUS.get(name)
            if g and (g.vram_gb < m.min_vram_gb or g.compute_cap < m.min_compute):
                raise typer.BadParameter(f"{name} ({g.vram_gb} GB, compute {g.compute_cap / 100:g}) is too small "
                                         f"for {m.key} (needs ≥{m.min_vram_gb} GB"
                                         + (f", compute ≥{m.min_compute / 100:g}" if m.min_compute else "") + ")")
            out.append(name)
    return list(dict.fromkeys(out))


def resolve_gpus(m: Model, given: Optional[List[str]], ask: bool, simulate: Optional[str] = None) -> List[str]:
    if given:
        return parse_gpus(m, given)
    if not ask:
        return list(m.default_gpus)
    console.print(gpu_table(m, simulate))
    default = ",".join(m.default_gpus) if m.default_gpus else "0"
    while True:
        try:
            return parse_gpus(m, [Prompt.ask("Cards (numbers/names, comma-separated; 0 = any)",
                                              default=default, console=console)])
        except typer.BadParameter as e:
            console.print(f"[red]{e}[/]")


def resolve_disk(m: Model, given: Optional[int], ask: bool) -> int:
    minimum = int(m.download_gb + 10)
    disk = given
    if disk is None and ask:
        disk = IntPrompt.ask(f"Disk GB (≥{minimum})", default=m.disk_gb, console=console)
    disk = disk or m.disk_gb
    if disk < minimum:
        raise typer.BadParameter(f"--disk {disk} GB is too small for {m.key}: needs ≥{minimum} GB "
                                 f"(~{m.download_gb:g} GB download + headroom)")
    return disk


def run_env(m: Model, gpus: List[str], disk: int, region: Optional[str],
            max_usd: Optional[float], max_hours: Optional[float]) -> dict:
    gpu_filter = m.base_filter()
    if gpus:
        gpu_filter += " gpu_name in [" + ",".join(gpus) + "]"
    env = dict(os.environ,
               RUN_MODEL=m.ref, RUN_GPU_FILTER=gpu_filter, RUN_DISK_GB=str(disk),
               RUN_DOWNLOAD_GB=f"{m.download_gb:g}", RUN_BENCH_HOURS=f"{m.bench_hours:g}", RUN_LABEL=m.key)
    if m.revision:
        env["RUN_REVISION"] = m.revision
    if m.vllm_preset:
        env["VLLM_PRESET"] = m.vllm_preset
    if region:
        env["REGION"] = region
    if max_usd is not None:
        env["MAX_RUN_USD_OLLAMA" if m.engine == "ollama" else "MAX_RUN_USD_VLLM"] = f"{max_usd:g}"
    if max_hours is not None:
        env["MAX_HOURS"] = f"{max_hours:g}"
    return env


def show_choice(m: Model, gpus: List[str], disk: int, region: Optional[str]) -> None:
    console.print(f"[bold]Run:[/] {m.key} ({m.engine}) · cards: {', '.join(gpus) or f'any ≥{m.min_vram_gb} GB'}"
                  f" · disk {disk} GB · region {region or os.environ.get('REGION', 'europe (default)')}")


def run_script(script: str, args: List[str], env: Optional[dict] = None) -> int:
    """Run a vast/*.sh script in the foreground. Ctrl-C goes to the script (its trap fetches
    logs and destroys the instance); we wait for it instead of dying first."""
    cmd = [str(VAST_DIR / script)] + args
    old = signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        proc = subprocess.Popen(cmd, env=env or os.environ.copy(), cwd=str(ROOT),
                                preexec_fn=lambda: signal.signal(signal.SIGINT, signal.SIG_DFL))
        return proc.wait()
    finally:
        signal.signal(signal.SIGINT, old)


def check_target_fit(m: Model, target: str, disk: int) -> None:
    t = TARGETS[target]
    need = m.weights_gb * 1.1 + 2
    console.print(f"[bold]Simulating[/] {t.title}: {m.key} needs ~{need:.0f} GB of {t.usable_gb} GB usable "
                  + ("[green]✓ fits[/]" if need <= t.usable_gb else "[red]✗ does not fit[/]")
                  + f"; disk {disk} GB of {t.ssd_gb} GB SSD " + ("[green]✓[/]" if disk <= t.ssd_gb else "[red]✗[/]"))
    if need > t.usable_gb:
        raise typer.BadParameter(f"{m.key} does not fit in {t.title}")
    console.print(f"  Benchmark runs on a rented card; afterwards {target}_projection.md converts the results "
                  f"(decode scaled by memory bandwidth, prefill by compute). Real GB10 hosts are rare on vast.ai and "
                  f"arm64-only, so they are not rented directly.")


def newest_run_since(start: float) -> Optional[Path]:
    runs = [p.parent for p in (ROOT / "output_vast").glob("*/llm_speed_*.json") if p.stat().st_mtime >= start - 5]
    return max(runs, key=lambda p: p.stat().st_mtime) if runs else None


def do_projection(run_dir: Path, target: str) -> None:
    try:
        res = project_run(str(run_dir), target)
    except (FileNotFoundError, ValueError) as e:
        console.print(f"[yellow]No {target} projection: {e}[/]")
        return
    path = write_report(str(run_dir), res, target)
    console.print(Path(path).read_text())
    console.print(f"[bold]Projection written:[/] {Path(path).relative_to(ROOT)}")


# Shared option definitions
SIM_OPT = typer.Option(None, "--simulate", "-s", help="Project the results onto a machine you can't rent: "
                       + ", ".join(f"{k} = {v.title}" for k, v in TARGETS.items()))
MODEL_ARG = typer.Argument(None, help="Model key, list number or unique prefix (see `models`). Asked if omitted.")
GPU_OPT = typer.Option(None, "--gpu", "-g", help="Card(s) to allow, e.g. -g 3090 -g 4090, or 'any'. Asked if omitted.")
DISK_OPT = typer.Option(None, "--disk", "-d", min=10, help="Disk GB (default: per model). Asked if omitted.")
REGION_OPT = typer.Option(None, "--region", "-r", help="europe (default) | eu | any | DE,PL,NL …")
NOINPUT_OPT = typer.Option(False, "--no-input", help="Never ask; use defaults for anything not given.")


# ----------------------------------------------------------------------------- commands
@app.command()
def models() -> None:
    """List the models that can be benchmarked."""
    console.print(model_table())
    console.print("Add more in [bold]vast/models.py[/].")


@app.command()
def cards(model: Optional[str] = MODEL_ARG,
          simulate: Optional[str] = typer.Option(None, "--simulate", "-s", help="Mark cards with bandwidth close to this target.")) -> None:
    """List the cards a model fits on (with memory bandwidth)."""
    console.print(gpu_table(resolve_model(model, ask=interactive(False)), simulate if simulate in TARGETS else None))


@app.command()
def offers(model: Optional[str] = MODEL_ARG, gpu: Optional[List[str]] = GPU_OPT,
           disk: Optional[int] = DISK_OPT, region: Optional[str] = REGION_OPT,
           interruptible: bool = typer.Option(False, "--interruptible", help="Price interruptible bids instead."),
           no_input: bool = NOINPUT_OPT) -> None:
    """Show the offers ranked by estimated total run cost — rents nothing."""
    ask = interactive(no_input)
    m = resolve_model(model, ask)
    gpus = resolve_gpus(m, gpu, ask)
    d = resolve_disk(m, disk, ask)
    show_choice(m, gpus, d, region)
    args = [m.engine, "--dry-run"] + (["--interruptible"] if interruptible else [])
    raise typer.Exit(run_script("launch.sh", args, run_env(m, gpus, d, region, None, None)))


@app.command()
def bench(model: Optional[str] = MODEL_ARG, gpu: Optional[List[str]] = GPU_OPT,
          disk: Optional[int] = DISK_OPT, region: Optional[str] = REGION_OPT,
          quick: bool = typer.Option(False, "--quick", "-q", help="Smoke-size benchmark (~2 min)."),
          interruptible: bool = typer.Option(False, "--interruptible", help="Bid instead of on-demand (can be stopped by vast)."),
          yes: bool = typer.Option(False, "--yes", "-y", help="No questions at all: defaults + rent without confirmation."),
          max_usd: Optional[float] = typer.Option(None, "--max-usd", help="Refuse offers estimated above this ($)."),
          max_hours: Optional[float] = typer.Option(None, "--max-hours", help="Hard stop for the whole run (h)."),
          simulate: Optional[str] = SIM_OPT,
          no_input: bool = NOINPUT_OPT) -> None:
    """Rent the cheapest suitable machine, benchmark, fetch results, destroy it."""
    if simulate and simulate not in TARGETS:
        raise typer.BadParameter(f"--simulate must be one of: {', '.join(TARGETS)}")
    ask = interactive(no_input or yes)
    m = resolve_model(model, ask)
    gpus = resolve_gpus(m, gpu, ask, simulate)
    d = resolve_disk(m, disk, ask)
    if simulate:
        check_target_fit(m, simulate, d)
    show_choice(m, gpus, d, region)
    args = [m.engine] + (["--quick"] if quick else []) + (["--yes"] if yes else []) \
        + (["--interruptible"] if interruptible else [])
    start = time.time()
    rc = run_script("bench.sh", args, run_env(m, gpus, d, region, max_usd, max_hours))
    if simulate:
        run = newest_run_since(start)
        if run:
            do_projection(run, simulate)
        else:
            console.print(f"[yellow]No benchmark results from this run — nothing to project onto {simulate}.[/]")
    raise typer.Exit(rc)


@app.command()
def project(instance_id: str = typer.Argument(..., help="Run in output_vast/ to project (see `results`)."),
            target: str = typer.Option("gx10", "--target", "-t", help="Machine to project onto: " + ", ".join(TARGETS))) -> None:
    """Project an existing run onto a machine you can't rent (default: ASUS Ascent GX10 / GB10)."""
    if target not in TARGETS:
        raise typer.BadParameter(f"--target must be one of: {', '.join(TARGETS)}")
    run = ROOT / "output_vast" / instance_id
    if not run.is_dir():
        raise typer.BadParameter(f"no {run}")
    do_projection(run, target)


@app.command()
def status() -> None:
    """Show the instance this checkout is tracking (if any)."""
    sid = VAST_DIR / ".instance_id"
    if not sid.exists():
        console.print("No instance tracked.")
        raise typer.Exit(0)
    iid = sid.read_text().strip()
    console.print(f"Tracked instance: [bold]{iid}[/]")
    raise typer.Exit(subprocess.call(["vastai", "show", "instance", iid]))


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def ssh(ctx: typer.Context) -> None:
    """SSH into the tracked instance, optionally running a command: ssh 'tail -f /workspace/setup.log'"""
    raise typer.Exit(run_script("ssh.sh", list(ctx.args)))


@app.command()
def fetch(instance_id: Optional[str] = typer.Argument(None, help="Default: the tracked instance.")) -> None:
    """Copy logs + reports into output_vast/<id>/ and write run_summary.md."""
    raise typer.Exit(run_script("fetch_results.sh", [instance_id] if instance_id else []))


@app.command()
def destroy(instance_id: Optional[str] = typer.Argument(None, help="Default: the tracked instance."),
            yes: bool = typer.Option(False, "--yes", "-y", help="Don't ask.")) -> None:
    """Destroy an instance and verify it is gone."""
    raise typer.Exit(run_script("destroy.sh", (["-y"] if yes else []) + ([instance_id] if instance_id else [])))


@app.command()
def results(instance_id: Optional[str] = typer.Argument(None, help="Show this run's summary; omit to list runs.")) -> None:
    """List past runs in output_vast/, or show one run's summary."""
    out = ROOT / "output_vast"
    if instance_id:
        p = out / instance_id / "run_summary.md"
        if not p.exists():
            raise typer.BadParameter(f"no {p}")
        console.print(p.read_text())
        for r in sorted((out / instance_id).glob("*_projection.md")):
            console.print(r.read_text())
        for r in sorted((out / instance_id).glob("llm_*.md")):
            console.print(f"  report: {r.relative_to(ROOT)}")
        return
    import json
    t = Table(title="Runs in output_vast/", title_justify="left")
    for col in ("instance", "model", "gpu", "outcome", "hours", "$"):
        t.add_column(col)
    for d in sorted(out.glob("*/run_summary.json"), key=lambda p: p.stat().st_mtime):
        s = json.loads(d.read_text())
        t.add_row(str(s.get("instance_id")), f"{s.get('mode')}-{s.get('preset')}", str(s.get("gpu_name")),
                  str(s.get("outcome")), str(s.get("billed_hours_so_far")), str(s.get("actual_usd_estimate")))
    console.print(t)


if __name__ == "__main__":
    app(prog_name="vastbench")
