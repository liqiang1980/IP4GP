import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import seaborn as sns


class TacPlotClass_dip():
    def __init__(self, tacperception):
        self._dataClass = tacperception
        self.fig = plt.figure(figsize=(20, 5))
        plt.subplot(141)
        plt.title("rf", fontsize=25)
        sns.heatmap(self._dataClass.tacdata_rf, vmin=-2, vmax=2, cbar=False, cbar_kws={}, cmap=sns.diverging_palette(10, 220, sep=80, n=7), annot=False,
                    annot_kws={"fontsize": 12}, fmt='g', yticklabels=False, xticklabels=False, linewidths=.5)
        plt.subplot(142)
        plt.title("mf", fontsize=25)
        sns.heatmap(self._dataClass.tacdata_mf, vmin=-2, vmax=2, cbar=False, cbar_kws={}, cmap=sns.diverging_palette(10, 220, sep=80, n=7), annot=False,
                    annot_kws={"fontsize": 12}, fmt='g', yticklabels=False, xticklabels=False, linewidths=.5)
        plt.subplot(143)
        plt.title("ff", fontsize=25)
        sns.heatmap(self._dataClass.tacdata_ff, vmin=-2, vmax=2, cbar=False, cbar_kws={}, cmap=sns.diverging_palette(10, 220, sep=80, n=7), annot=False,
                    annot_kws={"fontsize": 12}, fmt='g', yticklabels=False, xticklabels=False, linewidths=.5)
        plt.subplot(144)
        plt.title("th", fontsize=25)
        sns.heatmap(self._dataClass.tacdata_th, vmin=-2, vmax=2, cbar=False, cbar_kws={}, cmap=sns.diverging_palette(10, 220, sep=80, n=7), annot=False,
                    annot_kws={"fontsize": 12}, fmt='g', yticklabels=False, xticklabels=False, linewidths=.5)

        self.anim = animation.FuncAnimation(self.fig, func=self.animate, frames=None, save_count=0, interval=1,
                                            blit=False, repeat=True, cache_frame_data=False)

    def animate(self, i):
        plt.clf()
        # print("???")
        plt.subplot(141)
        plt.title("rf", fontsize=25)
        sns.heatmap(self._dataClass.tacdata_rf, vmin=-4, vmax=4, cbar=False, cbar_kws={}, cmap=sns.diverging_palette(10, 220, sep=80, n=7), annot=False,
                    annot_kws={"fontsize": 12}, fmt='g', yticklabels=False, xticklabels=False, linewidths=.5)
        plt.subplot(142)
        plt.title("mf", fontsize=25)
        sns.heatmap(self._dataClass.tacdata_mf, vmin=-4, vmax=4, cbar=False, cbar_kws={}, cmap=sns.diverging_palette(10, 220, sep=80, n=7), annot=False,
                    annot_kws={"fontsize": 12}, fmt='g', yticklabels=False, xticklabels=False, linewidths=.5)
        plt.subplot(143)
        plt.title("ff", fontsize=25)
        sns.heatmap(self._dataClass.tacdata_ff, vmin=-4, vmax=4, cbar=False, cbar_kws={}, cmap=sns.diverging_palette(10, 220, sep=80, n=7), annot=False,
                    annot_kws={"fontsize": 12}, fmt='g', yticklabels=False, xticklabels=False, linewidths=.5)
        plt.subplot(144)
        plt.title("th", fontsize=25)
        sns.heatmap(self._dataClass.tacdata_th, vmin=-4, vmax=4, cbar=False, cbar_kws={}, cmap=sns.diverging_palette(10, 220, sep=80, n=7), annot=False,
                    annot_kws={"fontsize": 12}, fmt='g', yticklabels=False, xticklabels=False, linewidths=.5)
        # sns.heatmap(self._dataClass.tacdata_th, vmin=0, vmax=2, cbar=False, cbar_kws={}, cmap="YlGnBu", annot=False,
        #             annot_kws={"fontsize": 12}, fmt='g', yticklabels=False, xticklabels=False, linewidths=.5)
        # print("11")
