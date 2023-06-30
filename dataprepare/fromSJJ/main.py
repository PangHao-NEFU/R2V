import subprocess
import os
import time
import sys


def startProcess():
    while True:
        print("守护进程开始")
        process = subprocess.Popen(['python', './FetchImg.py'])
        try:
            childpid = process.pid
            print("当前进程号pid" + str(childpid))
            process.wait()

            if process.returncode == 0:
                print("正常退出")
                sys.exit(0)
            print("异常退出,重启子进程")
            time.sleep(15)

        except subprocess.CalledProcessError as e:
            errorMeg = e.stderr.decode().strip()
            print("子进程发送异常", errorMeg)
            time.sleep(5)
        except Exception as e:
            print("其他异常", e)
            time.sleep(5)


if __name__ == '__main__':
    startProcess()
