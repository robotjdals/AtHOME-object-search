"""Operator console: natural-language command -> confirmed targets -> search.

    ros2 run athome_ros command_cli

1. The command is parsed by /athome/parse_command (the robot does not move).
2. The parsed targets are shown; the operator confirms (y), edits them (e) or
   drops the command (n). Human confirmation catches parsing errors before
   any search starts (human-in-the-loop, as in InteLiPlan).
3. Confirmed targets go to /athome/search_objects; progress is printed.
   Ctrl-C during a search cancels it (the robot stops first). A paused
   command can be resumed.
"""

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions

from athome_interfaces.action import SearchObjects
from athome_interfaces.srv import ParseCommand

_STATUS = {1: "완료", 2: "최대 탐색 횟수 도달", 3: "취소됨", 4: "일시중지"}


class CommandCli(Node):
    def __init__(self):
        super().__init__("athome_command_cli")
        self._parse = self.create_client(ParseCommand, "/athome/parse_command")
        self._search = ActionClient(self, SearchObjects, "/athome/search_objects")

    def wait_until(self, future, timeout=None):
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout)
        return future.result() if future.done() else None

    def parse(self, instruction):
        if not self._parse.wait_for_service(timeout_sec=5.0):
            print("명령 해석 서비스 없음 (search_server 실행 확인)")
            return None
        return self.wait_until(
            self._parse.call_async(ParseCommand.Request(instruction=instruction)), 30.0)

    def search(self, targets=(), resume=False):
        if not self._search.wait_for_server(timeout_sec=5.0):
            print("탐색 서버 없음")
            return None
        goal = SearchObjects.Goal(targets=list(targets), resume=resume)
        handle = self.wait_until(self._search.send_goal_async(goal, feedback_callback=_feedback), 10.0)
        if handle is None or not handle.accepted:
            print("탐색 요청 거절됨 (search_server 로그 확인)")
            return None
        result_future = handle.get_result_async()
        try:
            self.wait_until(result_future)
        except KeyboardInterrupt:
            print("\n취소 요청: 로봇 정지 확인 대기")
            self.wait_until(handle.cancel_goal_async(), 10.0)
            self.wait_until(result_future, 30.0)
        return result_future.result().result if result_future.done() else None


def _feedback(message):
    fb = message.feedback
    found = f", 발견: {list(fb.found_targets)}" if fb.found_targets else ""
    print(f"  step {fb.step} [{fb.target}] {fb.stage} → {fb.location_id} ({fb.path_cost:.1f} m){found}")


def _show(result):
    if result is None:
        print("결과 없음")
        return
    where = f" (위치 {result.location_id})" if result.location_id else ""
    print(f"결과: {_STATUS.get(result.status, result.status)} {result.reason} {result.detail}{where}")
    for t in result.targets:
        at = f" @ {t.found_location}" if t.found_location else ""
        print(f"  {t.name}: {t.status}{at} {t.detail}")


def _ask(prompt):
    return input(prompt).strip().lower()


def main(args=None):
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    cli = CommandCli()
    try:
        while True:
            try:
                instruction = input("\n명령 (빈 줄이면 종료)> ").strip()
            except EOFError:
                break
            if not instruction:
                break
            response = cli.parse(instruction)
            if response is None:
                print("명령 해석 응답 없음")
                continue
            if not response.success:
                print(f"찾을 물체 없음: {response.message}")
                continue
            for name, known in zip(response.targets, response.in_training_vocabulary):
                print(f"  - {name}" + ("" if known else "  (학습 어휘 밖)"))
            answer = _ask("이대로 찾을까요? [y: 예 / e: 수정 / n: 취소]> ")
            if answer == "e":
                edited = input("찾을 물체 (쉼표로 구분)> ")
                targets = [t.strip() for t in edited.split(",") if t.strip()]
            elif answer == "y":
                targets = list(response.targets)
            else:
                print("취소")
                continue
            if not targets:
                continue
            result = cli.search(targets)
            _show(result)
            while result is not None and result.status == 4 and _ask("원인을 해결했으면 재개할까요? [y/n]> ") == "y":
                result = cli.search(resume=True)
                _show(result)
    except KeyboardInterrupt:
        pass
    finally:
        cli.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
