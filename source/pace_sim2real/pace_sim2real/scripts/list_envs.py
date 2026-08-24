"""List PACE task IDs registered with mjlab."""

from __future__ import annotations

from prettytable import PrettyTable


def main() -> None:
    from mjlab.tasks.registry import list_tasks

    import pace_sim2real.tasks  # noqa: F401

    task_ids = [name for name in list_tasks() if "Pace" in name]
    table = PrettyTable(["#", "Task ID"])
    table.title = "Available PACE Environments in mjlab"
    table.align["Task ID"] = "l"
    for index, task_id in enumerate(task_ids, 1):
        table.add_row([index, task_id])
    print(table)


if __name__ == "__main__":
    main()
