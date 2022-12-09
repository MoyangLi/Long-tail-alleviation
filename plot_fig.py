import numpy as np

import matplotlib.pyplot as plt

with np.load('./FixMatch_result.npz') as Res1:
    FixMatch_Loss = Res1['Loss']
    FixMatch_Acc = Res1['Acc']

with np.load('./Fix_finetune_result.npz') as Res2:
    Fix_finetune_Loss = Res2['Loss']
    Fix_finetune_Acc = Res2['Acc']
x1 = range(np.size(FixMatch_Loss))
x2 = range(300, 360, 1)
plt.plot(x1, FixMatch_Loss)
plt.plot(x2, Fix_finetune_Loss)
plt.xlabel('epochs')
plt.ylabel('Loss')
plt.show()
plt.plot(x1, FixMatch_Acc)
plt.plot(x2, Fix_finetune_Acc)
plt.xlabel('epochs')
plt.ylabel('Accuracy')
plt.show()
