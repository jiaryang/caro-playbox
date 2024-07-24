import os
import signal

PID = os.getpid()

def do_nothing(*args):
    pass

def foo():
    print( "Initializing...")
    a=10
    os.kill(PID, signal.SIGUSR1)
    print( "Variable value is %d" % (a))
    print( "All done!")

signal.signal(signal.SIGUSR1, do_nothing)

foo()
