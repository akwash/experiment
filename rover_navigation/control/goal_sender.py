from __future__ import annotations

import argparse

from rover_navigation.control.goal_protocol import GoalCommand
from rover_navigation.control.goal_transport import UdpJsonGoalSender


def _read_int(prompt: str) -> int:
    """Read an integer from stdin with validation."""
    while True:
        raw = input(prompt).strip()
        if raw.lower() in {"q", "quit", "exit"}:
            raise KeyboardInterrupt
        try:
            return int(raw)
        except ValueError:
            print("Please enter a valid integer.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Terminal goal sender over UDP JSON.")
    parser.add_argument("--host", required=True, help="Jetson IP / hostname")
    parser.add_argument("--port", required=True, type=int, help="Jetson goal listener port")
    args = parser.parse_args()

    sender = UdpJsonGoalSender(host=args.host, port=args.port)
    print(f"Goal sender ready. Sending to {args.host}:{args.port}")
    print("Enter goal as (row, col). Type q to quit.\n")

    try:
        while True:
            row = _read_int("Enter goal row: ")
            col = _read_int("Enter goal col: ")

            cmd = GoalCommand(goal_cell=(row, col))
            sender.send_goal(cmd)
            print(f"Sent goal command: type={cmd.command_type} goal_cell=[{row},{col}]\n")
    except KeyboardInterrupt:
        print("\nExiting goal sender.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())

