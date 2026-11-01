import argparse
import json
import sys
from pathlib import Path

import yaml

from social_sim.authoring import load_scenario
from social_sim.diagnostics import explain, metrics
from social_sim.engine import Simulation
from social_sim.storage import export_run, load_snapshot


def natural(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError('Значение должно быть неотрицательным')
    return number


def fresh_output(path: Path) -> None:
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError(f'Каталог результата уже занят: {path}; укажите новый --output')


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Living School: причинный социальный симулятор')
    commands = parser.add_subparsers(dest='command', required=True)
    validate = commands.add_parser('validate', help='Проверить YAML-сценарии')
    validate.add_argument('path', type=Path)
    for name in ('run', 'batch'):
        sub = commands.add_parser(name)
        sub.add_argument('--scenario', type=Path, required=True)
        sub.add_argument('--days', type=natural, default=30)
        sub.add_argument('--seed', type=int, default=42)
        sub.add_argument('--output', type=Path)
        sub.add_argument('--no-player', action='store_true', help='Игрок бездействует (также поведение по умолчанию)')
        if name == 'batch':
            sub.add_argument('--count', type=natural, default=10)
    resume = commands.add_parser('resume')
    resume.add_argument('--snapshot', type=Path, required=True)
    resume.add_argument('--days', type=natural, required=True)
    resume.add_argument('--output', type=Path, required=True)
    explanation = commands.add_parser('explain')
    explanation.add_argument('--run', type=Path, required=True)
    explanation.add_argument('--event-id', required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == 'validate':
            files = sorted(args.path.rglob('*.yaml')) if args.path.is_dir() else [args.path]
            if not files:
                raise ValueError('Не найдено YAML-сценариев')
            for file in files:
                scenario = load_scenario(file)
                print(f'{file}: проверено {len(scenario.people)} участников')
            return 0
        if args.command == 'explain':
            simulation = load_snapshot(args.run / 'snapshot.json')
            print(json.dumps(explain(simulation.world, args.event_id), ensure_ascii=False, indent=2))
            return 0
        if args.command == 'batch':
            if args.count < 1:
                raise ValueError('Нужен хотя бы один прогон')
            scenario = load_scenario(args.scenario)
            directory = args.output or Path(f'output/batch_{args.seed}')
            fresh_output(directory)
            results = []
            for seed in range(args.seed, args.seed + args.count):
                simulation = Simulation(scenario, seed)
                simulation.run_days(args.days)
                # Same validation as a saved run, without retaining huge per-seed logs.
                type(simulation.world).model_validate(simulation.world.model_dump())
                results.append({'seed': seed, **metrics(simulation.world)})
            directory.mkdir(parents=True, exist_ok=True)
            report = json.dumps(results, ensure_ascii=False, indent=2)
            (directory / 'metrics.json').write_text(report, encoding='utf-8')
            print(report)
            return 0
        if args.command == 'resume':
            simulation = load_snapshot(args.snapshot)
            directory = args.output
        else:
            simulation = Simulation(load_scenario(args.scenario), args.seed)
            directory = args.output or Path(f'output/run_{args.seed}')
        fresh_output(directory)
        simulation.run_days(args.days)
        export_run(simulation, directory)
        report = json.dumps(metrics(simulation.world), ensure_ascii=False, indent=2)
        (directory / 'metrics.json').write_text(report, encoding='utf-8')
        print(report)
        print(f'Результаты: {directory}')
        return 0
    except (ValueError, OSError, yaml.YAMLError) as exc:
        print(f'Ошибка: {exc}', file=sys.stderr)
        return 2
