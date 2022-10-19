from threading import Thread

from time import sleep

import sys, termios, tty, os, time
 
def getch():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
 
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch
 
button_delay = 0.2

def function02(arg,name):
    while True:
        char = getch()
        #char = sys.stdin.read(1)
    
        if (char == "p"):
            print("Stop!")
            exit(0)
    
        if (char == "a"):
            print("Left pressed")
            time.sleep(button_delay)
    
        elif (char == "d"):
            print("Right pressed")
            time.sleep(button_delay)
    
        elif (char == "w"):
            print("Up pressed")
            time.sleep(button_delay)
    
        elif (char == "s"):
            print("Down pressed")
            time.sleep(button_delay)
    
        elif (char == "1"):
            print("Number 1 pressed")
            time.sleep(button_delay)

def function01(arg,name):
    for i in range(arg):
        print(name,'i---->',i,'\n')
        print (name,"arg---->",arg,'\n')
        sleep(1)

def test01():
    #thread1 = Thread(target = function01, args = (20,'thread1', ))
    #thread1.start()
    thread2 = Thread(target = function02, args = (50,'thread2', ))
    thread2.start()
    #thread1.join()
    
    for i in range(10):
        print ("main thread is running")
        sleep(0.5)
    thread2.join()
    print ("thread finished...exiting")

test01()
