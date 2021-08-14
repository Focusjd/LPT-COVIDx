from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


####object detection####
sns.set(
    context="paper",
    style="ticks",
    # palette="muted",
    palette="Paired",
    rc={
        # "pdf.fonttype": 42,
        "axes.labelsize": 18,
        # "axes.labelpad": 2.5,
        "xtick.labelsize": 10,
        # "xtick.major.pad": 0.0,
        "ytick.labelsize": 14,
        # "ytick.major.pad": 0.0,
        "legend.fontsize": 23,
        "axes.linewidth": 2.4,
        "text.latex.preamble": [
            r"\usepackage[tt=false, type1=true]{libertine}",
            r"\usepackage[varqu]{zi4}",
            r"\usepackage[libertine]{newtxmath}"]
    }
)




# # ax = plt.axes()
# colors = sns.color_palette()
# # colors = [colors[-2], colors[-1]]
# colors = [colors[-3]]
# pal = sns.color_palette("Reds_d", 5)
# # sns.set(font_scale=1.5, rc={'text.usetex': True})
# # sns.set(style="darkgrid")
# # plt.figure(figsize=(6.25,6))
# metrics = {r"$\gamma$": [0.1, 0.5, 1.0, 2.0, 3.0],
#            "error": [3.15, 2.75, 2.66, 3.07, 3.15]}
# rank = (np.asarray(metrics["error"])).argsort().argsort()

# metrics = pd.DataFrame(data=metrics)
# # plt.xticks(rotation=5)
# ax = sns.barplot(x=r"$\gamma$", y="error", data=metrics, palette=np.array(pal[::-1])[rank])
# ax.set(ylim=(2, None))
# # for index,row in metrics.iterrows():
# #         ax.text(row.method,row.mAP,round(row.mAP,2),color="black",ha="center")
# # show_values_on_bars(ax, "v", 0.3)
# # for p in ax.patches:
# #     ax.text(p.get_width(), p.get_y() + p.get_height()/2., '%d' % int(p.get_width()), 
# #             fontsize=12, color='red', ha='right', va='center')
# plt.title(r'The effect of $\gamma$ on CIFAR-10',fontsize=18)

# plt.savefig('./cifar-10-gamma.pdf', dpi=250)
# plt.show()


# # ax = plt.axes()
# colors = sns.color_palette()
# # colors = [colors[-2], colors[-1]]
# colors = [colors[-3]]
# pal = sns.color_palette("Reds_d", 5)
# # sns.set(font_scale=1.5, rc={'text.usetex': True})
# # sns.set(style="darkgrid")
# # plt.figure(figsize=(6.25,6))
# metrics = {r"$\gamma$": [0.1, 0.5, 1.0, 2.0, 3.0],
#            "error": [18.23, 16.00, 17.10, 17.50, 17.52]}
# rank = (np.asarray(metrics["error"])).argsort().argsort()

# metrics = pd.DataFrame(data=metrics)
# # plt.xticks(rotation=5)
# ax = sns.barplot(x=r"$\gamma$", y="error", data=metrics, palette=np.array(pal[::-1])[rank])
# ax.set(ylim=(15.5, None))
# # for index,row in metrics.iterrows():
# #         ax.text(row.method,row.mAP,round(row.mAP,2),color="black",ha="center")
# # show_values_on_bars(ax, "v", 0.3)
# # for p in ax.patches:
# #     ax.text(p.get_width(), p.get_y() + p.get_height()/2., '%d' % int(p.get_width()), 
# #             fontsize=12, color='red', ha='right', va='center')
# plt.title(r'The effect of $\gamma$ on CIFAR-100',fontsize=18)

# plt.savefig('./cifar-100-gamma.pdf', dpi=250)
# plt.show()

# # ax = plt.axes()
# colors = sns.color_palette()
# # colors = [colors[-2], colors[-1]]
# colors = [colors[-3]]
# pal = sns.color_palette("Blues_d", 5)
# # sns.set(font_scale=1.5, rc={'text.usetex': True})
# # sns.set(style="darkgrid")
# # plt.figure(figsize=(6.25,6))
# metrics = {r"$\lambda$": [0.1, 0.5, 1.0, 2.0, 3.0],
#            "error": [16.63, 17.23, 17.10, 17.09, 17.57]}
# rank = (np.asarray(metrics["error"])).argsort().argsort()

# metrics = pd.DataFrame(data=metrics)
# # plt.xticks(rotation=5)
# ax = sns.barplot(x=r"$\lambda$", y="error", data=metrics, palette=np.array(pal[::-1])[rank])
# # for index,row in metrics.iterrows():
# #         ax.text(row.method,row.mAP,round(row.mAP,2),color="black",ha="center")
# # show_values_on_bars(ax, "v", 0.3)
# # for p in ax.patches:
# #     ax.text(p.get_width(), p.get_y() + p.get_height()/2., '%d' % int(p.get_width()), 
# #             fontsize=12, color='red', ha='right', va='center')
# ax.set(ylim=(16, None))
# plt.title(r'The effect of $\lambda$ on CIFAR-100',fontsize=18)

# plt.savefig('./cifar-100-lambda.pdf', dpi=250)
# plt.show()


colors = sns.color_palette()
# colors = [colors[-2], colors[-1]]
colors = [colors[-3]]
pal = sns.color_palette("Blues_d", 5)
metrics = {r"$\lambda$": [0.1, 0.5, 1.0, 2.0, 3.0],
           "error": [2.67, 2.82, 2.66, 2.58, 2.88]}
rank = (np.asarray(metrics["error"])).argsort().argsort()

metrics = pd.DataFrame(data=metrics)
# plt.xticks(rotation=5)
ax = sns.barplot(x=r"$\lambda$", y="error", data=metrics, palette=np.array(pal[::-1])[rank])
ax.set(ylim=(2, None))
plt.title(r'The effect of $\lambda$ on CIFAR-10',fontsize=18)

plt.savefig('./cifar-10-lambda.pdf', dpi=250)
plt.show()