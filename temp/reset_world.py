#!/usr/bin/env python3
"""Call the Isaac world reset Trigger service once and exit."""

import argparse
import sys
import time

import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger


DEFAULT_SERVICE_NAME = "/world/reset"
DEFAULT_TIMEOUT_SEC = 30.0


class WorldResetClient(Node):
    def __init__(self, service_name: str) -> None:
        super().__init__("world_reset_client")
        self.client = self.create_client(Trigger, service_name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Send one std_srvs/Trigger request to reset the Isaac world."
    )
    parser.add_argument(
        "--service",
        default=DEFAULT_SERVICE_NAME,
        help=f"Reset service name. Default: {DEFAULT_SERVICE_NAME}",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT_SEC,
        help=f"Seconds to wait for the service and response. Default: {DEFAULT_TIMEOUT_SEC}",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    timeout_sec = max(float(args.timeout), 0.1)

    rclpy.init()
    node = WorldResetClient(args.service)
    deadline = time.monotonic() + timeout_sec
    try:
        while rclpy.ok() and not node.client.wait_for_service(timeout_sec=0.1):
            if time.monotonic() >= deadline:
                print(f"Timed out waiting for service: {args.service}", file=sys.stderr)
                return 1

        future = node.client.call_async(Trigger.Request())
        while rclpy.ok() and not future.done():
            if time.monotonic() >= deadline:
                print(f"Timed out waiting for response: {args.service}", file=sys.stderr)
                return 1
            rclpy.spin_once(node, timeout_sec=0.1)

        response = future.result()
        if response is None:
            print("Reset service returned no response.", file=sys.stderr)
            return 1

        print(f"success={response.success} message={response.message}")
        return 0 if response.success else 1
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
