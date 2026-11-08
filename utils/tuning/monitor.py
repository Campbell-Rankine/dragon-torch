"""Monitor optuna studies from the command line.

    python -m utils.tuning.monitor summary   --storage sqlite:///runs/optuna.db --study ppo-cartpole
    python -m utils.tuning.monitor export    --storage ... --study ... --out trials.csv
    python -m utils.tuning.monitor dashboard --storage ... [--host 127.0.0.1] [--port 8080]

`--storage` takes the same strings as TuneConfig.storage (including 'journal:<path>').
Omit `--study` in summary to list every study in the storage.
"""
import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Optional, Sequence, TextIO
import optuna
from .config import make_storage
from .study import best_summary


def trial_rows(study: optuna.Study) -> list[dict[str, Any]]:
    """One flat dict per trial: number, state, values, duration, params, last step."""
    rows = []
    for t in study.get_trials(deepcopy=False):
        row: dict[str, Any] = {'number': t.number, 'state': t.state.name}
        values = t.values or []
        for i, v in enumerate(values):
            row['value' if len(values) == 1 else f'value_{i}'] = v
        row['duration_s'] = round(t.duration.total_seconds(), 1) if t.duration else None
        row['last_step'] = t.last_step
        row.update({f'param_{k}': v for k, v in t.params.items()})
        rows.append(row)
    return rows


def _fmt(v: Any) -> str:
    if v is None:
        return '-'
    if isinstance(v, float):
        return f'{v:.4g}'
    return str(v)


def print_table(rows: Sequence[dict[str, Any]], out: Optional[TextIO] = None) -> None:
    out = out or sys.stdout
    if not rows:
        print('(no trials)', file=out)
        return
    cols = list(dict.fromkeys(k for r in rows for k in r))
    cells = [[_fmt(r.get(c)) for c in cols] for r in rows]
    widths = [max(len(c), *(len(row[i]) for row in cells)) for i, c in enumerate(cols)]
    print('  '.join(c.ljust(w) for c, w in zip(cols, widths)), file=out)
    for row in cells:
        print('  '.join(v.ljust(w) for v, w in zip(row, widths)), file=out)


def summarize(storage: Optional[str], study_name: Optional[str],
              out: Optional[TextIO] = None) -> None:
    out = out or sys.stdout
    store = make_storage(storage)
    if study_name is None:
        for s in optuna.get_all_study_summaries(store):
            best = s.best_trial.values if s.best_trial is not None else None
            print(f"{s.study_name}: {s.n_trials} trials, best={best}", file=out)
        return
    study = optuna.load_study(study_name=study_name, storage=store)
    rows = trial_rows(study)
    states = [r['state'] for r in rows]
    counts = ', '.join(f"{s.lower()}={states.count(s)}" for s in dict.fromkeys(states))
    print(f"study {study_name}  directions={[d.name.lower() for d in study.directions]}  "
          f"{counts or 'no trials'}", file=out)
    print_table(rows, out)
    best = best_summary(study)['best']
    if best:
        label = 'pareto front' if len(study.directions) > 1 else 'best'
        print(f"\n{label}:", file=out)
        for b in best:
            print(f"  #{b['number']} values={b['values']} params={b['params']}", file=out)


def export_csv(storage: Optional[str], study_name: str, path: Path) -> Path:
    rows = trial_rows(optuna.load_study(study_name=study_name, storage=make_storage(storage)))
    cols = list(dict.fromkeys(k for r in rows for k in r))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)
    return path


def dashboard(storage: str, host: str = '127.0.0.1', port: int = 8080) -> None:
    try:
        from optuna_dashboard import run_server
    except ImportError as e:
        raise SystemExit("optuna-dashboard is not installed: uv sync --extra tune") from e
    print(f"optuna-dashboard on http://{host}:{port}")
    run_server(make_storage(storage), host=host, port=port)


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(prog='python -m utils.tuning.monitor', description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='cmd', required=True)

    p = sub.add_parser('summary', help='print trials and the best result')
    p.add_argument('--storage', required=True)
    p.add_argument('--study', default=None)

    p = sub.add_parser('export', help='write all trials to CSV')
    p.add_argument('--storage', required=True)
    p.add_argument('--study', required=True)
    p.add_argument('--out', type=Path, default=Path('trials.csv'))

    p = sub.add_parser('dashboard', help='serve optuna-dashboard for the storage')
    p.add_argument('--storage', required=True)
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--port', type=int, default=8080)

    args = parser.parse_args(argv)
    if args.cmd == 'summary':
        summarize(args.storage, args.study)
    elif args.cmd == 'export':
        print(export_csv(args.storage, args.study, args.out))
    else:
        dashboard(args.storage, args.host, args.port)


if __name__ == '__main__':
    main()
