import time
while True:
    time.sleep(1800)
    os.system('say "The time is now " + time.strftime("%H:%M", time.localtime()))