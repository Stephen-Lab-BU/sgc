from glob import glob
import os

def run():
    cwd = os.getcwd()

    for x in glob(os.path.join(cwd, '*bash.o*')):
        os.remove(x)

    for x in glob(os.path.join(cwd, '*bash.e*')):
        os.remove(x)


if __name__ == "__main__":
    run()