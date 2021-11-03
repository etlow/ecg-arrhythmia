import matplotlib.pyplot as plt
import numpy as np
classes = ['Normal sinus', 'Atrial fibrillation', 'Atrial flutter', 'Ventricular bigeminy', 'Ventricular trigeminy', 'Noise']
freq = [[12795, 7527, 929, 2371, 610, 1688], [3344, 4853, 92, 276, 621, 294]]
names = ['Training set', 'Testing set']
fig, axs = plt.subplots(2)
for name, f, ax in zip(names, freq, axs):
    ax.bar(classes, f)
    ax.set_title(name)
#axs[0][0].legend()
fig.set_size_inches(fig.get_size_inches() * 1.7)
fig.tight_layout()
plt.show()
