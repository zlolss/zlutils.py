
import threading
import logging
import traceback


class Restartable:
    # 可重启的线程类

    def __init__(self):
        self.__thread: threading.Thread = None
        self.__running = False
        self.__run_count = 0

    def runstep(self):
        # 按步骤 循环执行 重写这部分
        1/0
        self.__running = False
        pass

    def run(self):
        # 一次性执行 重写这部分
        try:
            self.__running = True
            self.__run_count += 1
            print(f'{self.__running}运行{self.__run_count}次')
            while self.__running:
                self.runstep()
        except Exception as e:
            logging.error(f'{e}\n{traceback.format_exc()}')  # 输出完整错误日志
        finally:
            self.__running = False

    def stop(self):  # 此操作会产生阻塞
        self.__running = False
        if self.__thread.is_alive():
            self.__thread.join()

    def start(self):
        if self.__thread and self.__thread.is_alive():
            # already run
            return
        self.__thread = threading.Thread(target=self.run)
        self.__running = True
        self.__thread.start()

    def restart(self):  # 此操作会产生阻塞
        self.stop()
        self.start()


if __name__ == "__main__":
    rt = Restartable()
    for i in range(3):
        rt.start()
