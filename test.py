import time, random
from atpbar import atpbar

for i in atpbar(range(4), name='outer'):
    n = random.randint(1000, 10000)
    for j in atpbar(range(n), name='inner {}'.format(i)):
        time.sleep(0.0001)