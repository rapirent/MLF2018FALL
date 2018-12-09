import numpy as np
import random
from datetime import datetime

list_gogo = []
random.seed(datetime.now())
noise = np.random.choice([1,0],10000,[0.8,0.2])
# list_gogo.append(noise[0])

print(sum(noise)/10000)
