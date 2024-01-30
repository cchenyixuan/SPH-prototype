import numpy as np
import matplotlib.pyplot as plt


exp = []
with open(r"D:\ProgramFiles\PycharmProject\SPH-prototype\HE/exp00_truevolume.txt", "r") as f:
    for row in f:
        exp.append(row[:-1].split(" "))
    f.close()
exp = np.array(exp, dtype=np.float32)

ans = []
with open(r"D:\ProgramFiles\PycharmProject\SPH-prototype\HE/mp_ans00_wend_acc1.0.txt", "r") as f:
    for row in f:
        ans.append(row[:-1].split(" "))
    f.close()
ans = np.array(ans, dtype=np.float32)
fig, ax = plt.subplots()
ax.plot(exp[:, 1], exp[:, 0], color="#f58f29", label="SPH")
ax.scatter(ans[::1, 1], ans[::1, 0], c="#a4b0f5", label="Analytical")
ax.legend(["SPH u(0, t)", "Analytical"])
plt.xlabel("t")
plt.ylabel("u")
plt.show()

delta = []
ratio = []
ptr = 0
for i in range(ans.shape[0]):
    t = ans[i, 1]

    for j in range(ptr, exp.shape[0]):
        if abs(exp[j, 1]-t) < 5e-07:
            delta.append([abs(exp[j, 0] - ans[i, 0]), t])
            ratio.append([(exp[j, 0]-ans[i, 0]) / ans[i, 0], t])
            ptr = j
            break
delta = np.array(delta, dtype=np.float32)
ratio = np.array(ratio, dtype=np.float32)
plt.yscale("log")
plt.plot(delta[:, 1], delta[:, 0])
plt.plot(delta[:, 1], [0.1+0.000005**3*i*100*10 for i in range(delta.shape[0])])
plt.plot(delta[:, 1], [0.1*0.1+0.000005**3*i*100 for i in range(delta.shape[0])])
plt.legend(["|<u(0, t)>-u(0, t)|", "0.1", "0.01"])
plt.xlabel("t")
plt.ylabel("Error")
plt.show()

plt.plot(ratio[:, 1], abs(ratio[:, 0])*100)#-delta[1:-1, 0]+delta[:-2, 0])
plt.legend(["|<u(0, t)>-u(0, t)|/u(0, t)"])
plt.xlabel("t")
plt.ylabel("Error%")
plt.show()
