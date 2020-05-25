import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from pylab import *
mpl.rcParams['font.sans-serif'] = ['SimHei']

#加载预期输出和实际输出
y_pre=np.load(r'D:\pycharm_community\python_workspace\drug_classification/y_pred_bays.npy')  #预期输出 ''中换自己文件的路径
y_actual=np.load(r'D:\pycharm_community\python_workspace\drug_classification/y_true_bays.npy')  #实际输出
y_pre=np.array(y_pre)
y_actual=np.array(y_actual)

tp=[]
tn=[]
# for i in range(len(y_pre)):
#     if y_pre[i]==0 and y_actual[i]==0:
#         tn.append(1)
#     elif y_pre[i]==1 and y_actual[i]==1:
#         tp.append(1)
# tn_acc=len(tn)/len(y_pre)
# tp_acc=len(tp)/len(y_pre)
# print('0类识别准确率为：%.2f'%tn_acc)
# print('1类识别准确率为：%.2f'%tp_acc)

fpr,tpr,thresholds= metrics.roc_curve(y_actual.ravel(), y_pre.ravel())

# 插值法之后的fpr轴值，表示从0到1间距为0.1的100个数
fpr_new = np.arange(0,1,0.01)
# 实现函数
from scipy import interpolate
func = interpolate.interp1d(fpr,tpr, kind='slinear')
# 利用fpr_new和func函数生成tpr_new,fpr_new数量等于tpr_new数量
tpr_new = func(fpr_new )
auc=metrics.auc(fpr_new,tpr_new)
plt.plot(fpr_new,tpr_new,c='r',lw=1,alpha=0.7,label=u'AUC=%.3f'%auc)
plt.plot((0, 1),(0, 1),c='#808080',lw=1, ls='--',alpha=0.7)
plt.xlim((-0.01,1.02))
plt.ylim((-0.01,1.02))
plt.xticks(np.arange(0,1.1,0.1))
plt.yticks(np.arange(0,1.1,0.1))
plt.xlabel('False Positive Rate',fontsize=13)
plt.ylabel('True Positive Rate',fontsize=13)
plt.grid(b=True,ls=':')
plt.legend(loc='best',fancybox=True, framealpha=0.8, fontsize=12)
plt.title(u'ROC and AUC',fontsize=17)
plt.show()
