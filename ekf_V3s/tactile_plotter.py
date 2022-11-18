import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import seaborn as sns
from threading import Thread


# import test


class HeatmapAnimate_Dip:
    def __init__(self, tacperception):
        self._dataClass = tacperception
        # self.fig = plt.figure(figsize=(20, 5))
        self.fig, self.ax = plt.subplots(1, 4, figsize=(20, 5))
        # self.ax[0] = self.fig.add_subplot(1, 4, 1)
        # self.ax[1] = self.fig.add_subplot(1, 4, 2)
        # self.ax[2] = self.fig.add_subplot(1, 4, 3)
        # self.ax[3] = self.fig.add_subplot(1, 4, 4)
        # self.ax1 = self.ax[0]
        # self.ax2 = self.ax[1]
        # self.ax3 = self.ax[2]
        # self.ax4 = self.ax[3]
        self.plt_mapping = {0: [141, self._dataClass.tacdata_rf],
                            1: [142, self._dataClass.tacdata_mf],
                            2: [143, self._dataClass.tacdata_ff],
                            3: [144, self._dataClass.tacdata_th]}
        self.titles = ["rf", "mf", "ff", "th"]
        # self.plt_mapping = {0: [self.ax1, self._dataClass.tacdata_rf],
        #                     1: [self.ax2, self._dataClass.tacdata_mf],
        #                     2: [self.ax3, self._dataClass.tacdata_ff],
        #                     3: [self.ax4, self._dataClass.tacdata_th]}
        for i in range(4):
            plt.subplot(self.plt_mapping[i][0])
            plt.title(self.titles[i], fontsize=25)
            sns.heatmap(self.plt_mapping[i][1], vmin=-4, vmax=4, cbar=False, cbar_kws={},
                                     cmap=sns.diverging_palette(10, 220, sep=80, n=7), annot=False,
                                     annot_kws={"fontsize": 12}, fmt='g', yticklabels=False, xticklabels=False,
                                     linewidths=.5)
        self.anim = animation.FuncAnimation(self.fig, func=self.animate, frames=None, save_count=0, interval=1,
                                            blit=False, repeat=True, cache_frame_data=False)
        # plt.draw()

    def animate(self, i):
        plt.clf()
        for i in range(4):
            plt.subplot(self.plt_mapping[i][0])
            plt.title(self.titles[i], fontsize=25)
            sns.heatmap(self.plt_mapping[i][1], vmin=-4, vmax=4, cbar=False, cbar_kws={},
                                     cmap=sns.diverging_palette(10, 220, sep=80, n=7), annot=False,
                                     annot_kws={"fontsize": 12}, fmt='g', yticklabels=False, xticklabels=False,
                                     linewidths=.5)


class LineChartAnimate_Obj:
    def __init__(self, robo, x_len, y_len, label1, label2, label3, label4, label5, label6):
        """
        Plot a line chart animation for ekf_obj's gd and x_state.
        """
        self._dataClass = robo
        self.x_len = x_len
        # x_state = self._dataClass.x_state_cur
        # gd = self._dataClass.gd_cur
        # print("   check x_state, gd:", x_state, gd)
        """ Data Preprocess"""
        # self.x_state1 = x_state[0] * 1000
        # self.x_state2 = x_state[1] * 1000
        # self.x_state3 = x_state[2] * 1000
        # self.x_state4 = x_state[3] * 57.3
        # self.x_state5 = x_state[4] * 57.3
        # self.x_state6 = x_state[5] * 57.3
        # self.gd1 = gd[0] * 1000
        # self.gd2 = gd[1] * 1000
        # self.gd3 = gd[2] * 1000
        # self.gd4 = gd[3] * 57.3
        # self.gd5 = gd[4] * 57.3
        # self.gd6 = gd[5] * 57.3
        # self.x_state1 = x_state[:, 0] * 1000
        # self.x_state2 = x_state[:, 1] * 1000
        # self.x_state3 = x_state[:, 2] * 1000
        # self.x_state4 = x_state[:, 3] * 57.3
        # self.x_state5 = x_state[:, 4] * 57.3
        # self.x_state6 = x_state[:, 5] * 57.3
        # self.gd1 = gd[:, 0] * 1000
        # self.gd2 = gd[:, 1] * 1000
        # self.gd3 = gd[:, 2] * 1000
        # self.gd4 = gd[:, 3] * 57.3
        # self.gd5 = gd[:, 4] * 57.3
        # self.gd6 = gd[:, 5] * 57.3
        """ Set basic params """
        self.x_val = []
        self.y_val11, self.y_val12, self.y_val21, self.y_val22, \
        self.y_val31, self.y_val32, self.y_val41, self.y_val42, \
        self.y_val51, self.y_val52, self.y_val61, self.y_val62 = [], [], [], [], [], [], [], [], [], [], [], []
        sns.set_style('darkgrid')
        # self.t = np.arange(0, frame_len, 1)
        # self.fig = plt.figure(figsize=(12, 6))
        self.fig, ax = plt.subplots(3, 2, figsize=(12, 6), sharex='all')
        self.fig.subplots_adjust(hspace=0.1, wspace=0.2)
        # ax.set_title('')
        # ax.set_xlabel('x')
        # ax.set_ylabel('y')
        """ Plot Init """
        self.ax1 = ax[0, 0]
        self.ax1.legend(['y1, y2'], loc='upper left')
        self.ax1.set_ylabel(label1, {'size': 8})
        self.ax1.set_ylim(-(y_len + 1), (y_len + 1))
        self.ln11, = self.ax1.plot([], [], animated=False, color='red')
        self.ln12, = self.ax1.plot([], [], animated=False, color='black')
        self.ax2 = ax[1, 0]
        self.ax2.set_ylabel(label2, {'size': 8})
        self.ax2.set_ylim(-(y_len + 1), (y_len + 1))
        self.ln21, = self.ax2.plot([], [], animated=False, color='red')
        self.ln22, = self.ax2.plot([], [], animated=False, color='black')
        self.ax3 = ax[2, 0]
        self.ax3.set_ylabel(label3, {'size': 8})
        self.ax3.set_xlabel('Count', {'size': 8})
        self.ax3.set_ylim(-(y_len + 1), (y_len + 1))
        self.ax3.set_xlim(0, x_len + 1)
        self.ln31, = self.ax3.plot([], [], animated=False, color='red')
        self.ln32, = self.ax3.plot([], [], animated=False, color='black')
        self.ax4 = ax[0, 1]
        self.ax4.set_ylabel(label4, {'size': 8})
        self.ax4.set_ylim(-(y_len + 1), (y_len + 1))
        self.ln41, = self.ax4.plot([], [], animated=False, color='red')
        self.ln42, = self.ax4.plot([], [], animated=False, color='black')
        self.ax5 = ax[1, 1]
        self.ax5.set_ylabel(label5, {'size': 8})
        self.ax5.set_ylim(-(y_len + 1), (y_len + 1))
        self.ln51, = self.ax5.plot([], [], animated=False, color='red')
        self.ln52, = self.ax5.plot([], [], animated=False, color='black')
        self.ax6 = ax[2, 1]
        self.ax6.set_ylabel(label6, {'size': 8})
        self.ax6.set_xlabel('Count', {'size': 8})
        self.ax6.set_ylim(-(y_len + 1), (y_len + 1))
        self.ax6.set_xlim(0, x_len + 1)
        self.ln61, = self.ax6.plot([], [], animated=False, color='red')
        self.ln62, = self.ax6.plot([], [], animated=False, color='black')
        # self.anim = animation.FuncAnimation(self.fig, func=self.animate, frames=self.x_len + 1, save_count=0,
        #                                     interval=300, blit=False)
        self.anim = animation.FuncAnimation(self.fig, func=self.animate, frames=None, save_count=0, interval=1,
                                            blit=False, repeat=True, cache_frame_data=False)

    def animate(self, frame):
        """
        Update date and Draw one picture.
        """
        """ Data Preprocess"""
        # x_state = test.x_state_cur
        # gd = test.gd_cur
        x_state = self._dataClass.x_state_cur
        gd = self._dataClass.gd_cur
        self.x_state1 = x_state[0] * 1000
        self.x_state2 = x_state[1] * 1000
        self.x_state3 = x_state[2] * 1000
        self.x_state4 = x_state[3] * 57.3
        self.x_state5 = x_state[4] * 57.3
        self.x_state6 = x_state[5] * 57.3
        self.gd1 = gd[0] * 1000
        self.gd2 = gd[1] * 1000
        self.gd3 = gd[2] * 1000
        self.gd4 = gd[3] * 57.3
        self.gd5 = gd[4] * 57.3
        self.gd6 = gd[5] * 57.3
        # self.x_state1 = x_state[:, 0] * 1000
        # self.x_state2 = x_state[:, 1] * 1000
        # self.x_state3 = x_state[:, 2] * 1000
        # self.x_state4 = x_state[:, 3] * 57.3
        # self.x_state5 = x_state[:, 4] * 57.3
        # self.x_state6 = x_state[:, 5] * 57.3
        # self.gd1 = gd[:, 0] * 1000
        # self.gd2 = gd[:, 1] * 1000
        # self.gd3 = gd[:, 2] * 1000
        # self.gd4 = gd[:, 3] * 57.3
        # self.gd5 = gd[:, 4] * 57.3
        # self.gd6 = gd[:, 5] * 57.3
        # print("++++++update plot!++++++\n  ", self.x_val, "\n  ", self.y_val11, "\n  ", self.y_val12, "\n  ",
        #       self.y_val21, "\n  ", self.y_val22, "\n  ", self.y_val31, "\n  ", self.y_val32, "\n  ")
        self.x_val.append(frame)
        self.y_val11.append(self.x_state1)
        self.y_val12.append(self.gd1)
        self.y_val21.append(self.x_state2)
        self.y_val22.append(self.gd2)
        self.y_val31.append(self.x_state3)
        self.y_val32.append(self.gd3)
        self.y_val41.append(self.x_state4)
        self.y_val42.append(self.gd4)
        self.y_val51.append(self.x_state5)
        self.y_val52.append(self.gd5)
        self.y_val61.append(self.x_state6)
        self.y_val62.append(self.gd6)
        """ Update x & y axis """
        xmin1, xmax1 = self.ax1.get_xlim()
        ymin1, ymax1 = self.ax1.get_ylim()
        xmin2, xmax2 = self.ax2.get_xlim()
        ymin2, ymax2 = self.ax2.get_ylim()
        xmin3, xmax3 = self.ax3.get_xlim()
        ymin3, ymax3 = self.ax3.get_ylim()
        xmin4, xmax4 = self.ax4.get_xlim()
        ymin4, ymax4 = self.ax4.get_ylim()
        xmin5, xmax5 = self.ax5.get_xlim()
        ymin5, ymax5 = self.ax5.get_ylim()
        xmin6, xmax6 = self.ax6.get_xlim()
        ymin6, ymax6 = self.ax6.get_ylim()
        if len(self.x_val) > xmax1:
            self.ax1.set_xlim(xmin1, 2 * xmax1)
            self.ax1.figure.canvas.draw()
            self.ax2.set_xlim(xmin1, 2 * xmax1)
            self.ax2.figure.canvas.draw()
            self.ax3.set_xlim(xmin1, 2 * xmax1)
            self.ax3.figure.canvas.draw()
            self.ax4.set_xlim(xmin1, 2 * xmax1)
            self.ax4.figure.canvas.draw()
            self.ax5.set_xlim(xmin1, 2 * xmax1)
            self.ax5.figure.canvas.draw()
            self.ax6.set_xlim(xmin1, 2 * xmax1)
            self.ax6.figure.canvas.draw()
        if min(self.x_state1, self.gd1) < ymin1:
            self.ax1.set_ylim(2 * ymin1, ymax1)
        elif max(self.x_state1, self.gd1) > ymax1:
            self.ax1.set_ylim(ymin1, 2 * ymax1)
        if min(self.x_state2, self.gd2) < ymin2:
            self.ax2.set_ylim(2 * ymin2, ymax2)
        elif max(self.x_state2, self.gd2) > ymax2:
            self.ax2.set_ylim(ymin2, 2 * ymax2)
        if min(self.x_state3, self.gd3) < ymin3:
            self.ax3.set_ylim(2 * ymin3, ymax3)
        elif max(self.x_state3, self.gd3) > ymax3:
            self.ax3.set_ylim(ymin3, 2 * ymax3)
        if min(self.x_state4, self.gd4) < ymin4:
            self.ax4.set_ylim(2 * ymin4, ymax4)
        elif max(self.x_state4, self.gd4) > ymax4:
            self.ax4.set_ylim(ymin4, 2 * ymax4)
        if min(self.x_state5, self.gd5) < ymin5:
            self.ax5.set_ylim(2 * ymin5, ymax5)
        elif max(self.x_state5, self.gd5) > ymax5:
            self.ax5.set_ylim(ymin5, 2 * ymax5)
        if min(self.x_state6, self.gd6) < ymin6:
            self.ax6.set_ylim(2 * ymin6, ymax6)
        elif max(self.x_state6, self.gd6) > ymax6:
            self.ax6.set_ylim(ymin6, 2 * ymax6)

        """ Update ln, """
        self.ln11.set_data(self.x_val, self.y_val11)
        self.ln12.set_data(self.x_val, self.y_val12)
        self.ln21.set_data(self.x_val, self.y_val21)
        self.ln22.set_data(self.x_val, self.y_val22)
        self.ln31.set_data(self.x_val, self.y_val31)
        self.ln32.set_data(self.x_val, self.y_val32)
        self.ln41.set_data(self.x_val, self.y_val41)
        self.ln42.set_data(self.x_val, self.y_val42)
        self.ln51.set_data(self.x_val, self.y_val51)
        self.ln52.set_data(self.x_val, self.y_val52)
        self.ln61.set_data(self.x_val, self.y_val61)
        self.ln62.set_data(self.x_val, self.y_val62)
        # Attention: Add commas to convert the returned params to tuples containing the params
        return self.ln11, self.ln12, self.ln21, self.ln22, self.ln31, self.ln32, self.ln41, self.ln42, self.ln51, self.ln52, self.ln61, self.ln62,


class AllAnimate:
    def __init__(self, tacperception, robo, x_len, y_len, label1, label2, label3, label4, label5, label6):
        """
        Plot Heatmap animation and LineChart animation.
        """
        """
        Plot a heatmap animation for tac data in dips.
        """
        self._dataClass_tac = tacperception
        self.fig = plt.figure(figsize=(12, 8))
        sns.set_style('darkgrid')
        # self.fig, ax = plt.subplots(4, 4, figsize=(32, 6), sharex='all')
        # ax = self.fig.add_subplot(3, 2, figsize=(12, 6), sharex='all')
        self.fig.subplots_adjust(hspace=0.2, wspace=0.3)
        # self.plt_mapping = {0: [441, self._dataClass_tac.tacdata_rf],
        #                     1: [442, self._dataClass_tac.tacdata_mf],
        #                     2: [443, self._dataClass_tac.tacdata_ff],
        #                     3: [444, self._dataClass_tac.tacdata_th]}
        self.plt_mapping = {0: [(1, 5), self._dataClass_tac.tacdata_rf],
                            1: [(2, 6), self._dataClass_tac.tacdata_mf],
                            2: [(3, 7), self._dataClass_tac.tacdata_ff],
                            3: [(4, 8), self._dataClass_tac.tacdata_th]}
        self.titles = ["rf", "mf", "ff", "th"]
        for i in range(4):
            # plt.subplot(self.plt_mapping[i][0])
            plt.subplot(5, 4, self.plt_mapping[i][0])
            plt.title(self.titles[i], fontsize=15)
            sns.heatmap(self.plt_mapping[i][1], vmin=-4, vmax=4, cbar=False, cbar_kws={},
                        cmap=sns.diverging_palette(10, 220, sep=80, n=7), annot=False,
                        annot_kws={"fontsize": 12}, fmt='g', yticklabels=False, xticklabels=False, linewidths=.5)
        """
        Plot a line chart animation for ekf_obj's gd and x_state.
        """
        self._dataClass_rob = robo
        self.x_len = x_len
        """ Set basic params """
        self.x_val = []
        self.y_val11, self.y_val12, self.y_val21, self.y_val22, \
        self.y_val31, self.y_val32, self.y_val41, self.y_val42, \
        self.y_val51, self.y_val52, self.y_val61, self.y_val62 = [], [], [], [], [], [], [], [], [], [], [], []
        # sns.set_style('darkgrid')
        # self.fig, ax = plt.subplots(3, 2, figsize=(12, 6), sharex='all')
        # self.fig.subplots_adjust(hspace=0.1, wspace=0.2)
        """ Plot Init """
        # self.ax1 = ax[1, 0]
        self.ax1 = self.fig.add_subplot(5, 4, (9, 10))
        self.ax1.set_ylabel(label1, {'size': 8})
        self.ax1.set_ylim(-(y_len + 1), (y_len + 1))
        self.ln11, = self.ax1.plot([], [], animated=False, color='red')
        self.ln12, = self.ax1.plot([], [], animated=False, color='black')
        # plt.tick_params(labelsize=5)
        # labels = self.ax1.get_xticklabels() + self.ax1.get_yticklabels()
        # [label.set_fontname('Times New Roman') for label in labels]  # 设置坐标轴刻度值的大小和字体
        # self.ax2 = ax[2, 0]
        self.ax2 = self.fig.add_subplot(5, 4, (13, 14))
        self.ax2.set_ylabel(label2, {'size': 8})
        self.ax2.set_ylim(-(y_len + 1), (y_len + 1))
        self.ln21, = self.ax2.plot([], [], animated=False, color='red')
        self.ln22, = self.ax2.plot([], [], animated=False, color='black')
        # self.ax3 = ax[3, 0]
        self.ax3 = self.fig.add_subplot(5, 4, (17, 18))
        self.ax3.set_ylabel(label3, {'size': 8})
        self.ax3.set_xlabel('Count', {'size': 8})
        self.ax3.set_ylim(-(y_len + 1), (y_len + 1))
        self.ax3.set_xlim(0, x_len + 1)
        self.ln31, = self.ax3.plot([], [], animated=False, color='red')
        self.ln32, = self.ax3.plot([], [], animated=False, color='black')
        # self.ax4 = ax[0, 1]
        self.ax4 = self.fig.add_subplot(5, 4, (11, 12))
        self.ax4.set_ylabel(label4, {'size': 8})
        self.ax4.set_ylim(-(y_len + 1), (y_len + 1))
        self.ln41, = self.ax4.plot([], [], animated=False, color='red')
        self.ln42, = self.ax4.plot([], [], animated=False, color='black')
        # self.ax5 = ax[1, 1]
        self.ax5 = self.fig.add_subplot(5, 4, (15, 16))
        self.ax5.set_ylabel(label5, {'size': 8})
        self.ax5.set_ylim(-(y_len + 1), (y_len + 1))
        self.ln51, = self.ax5.plot([], [], animated=False, color='red')
        self.ln52, = self.ax5.plot([], [], animated=False, color='black')
        # self.ax6 = ax[2, 1]
        self.ax6 = self.fig.add_subplot(5, 4, (19, 20))
        self.ax6.set_ylabel(label6, {'size': 8})
        self.ax6.set_xlabel('Count', {'size': 8})
        self.ax6.set_ylim(-(y_len + 1), (y_len + 1))
        self.ax6.set_xlim(0, x_len + 1)
        self.ln61, = self.ax6.plot([], [], animated=False, color='red')
        self.ln62, = self.ax6.plot([], [], animated=False, color='black')

        self.anim = animation.FuncAnimation(self.fig, func=self.animate_all, frames=None, save_count=0, interval=1,
                                            blit=False, repeat=True, cache_frame_data=False)

    def animate_all(self, frame):
        """
        Update date and Draw one picture.
        """
        # plt.clf()
        for i in range(4):
            plt.subplot(5, 4, self.plt_mapping[i][0])
            plt.title(self.titles[i], fontsize=15)
            sns.heatmap(self.plt_mapping[i][1], vmin=-4, vmax=4, cbar=False, cbar_kws={},
                        cmap=sns.diverging_palette(10, 220, sep=80, n=7), annot=False,
                        annot_kws={"fontsize": 12}, fmt='g', yticklabels=False, xticklabels=False, linewidths=.5)
        """ Data Preprocess"""
        x_state = self._dataClass_rob.x_state_cur
        gd = self._dataClass_rob.gd_cur
        self.x_state1 = x_state[0] * 1000
        self.x_state2 = x_state[1] * 1000
        self.x_state3 = x_state[2] * 1000
        self.x_state4 = x_state[3] * 57.3
        self.x_state5 = x_state[4] * 57.3
        self.x_state6 = x_state[5] * 57.3
        self.gd1 = gd[0] * 1000
        self.gd2 = gd[1] * 1000
        self.gd3 = gd[2] * 1000
        self.gd4 = gd[3] * 57.3
        self.gd5 = gd[4] * 57.3
        self.gd6 = gd[5] * 57.3
        self.x_val.append(frame)
        self.y_val11.append(self.x_state1)
        self.y_val12.append(self.gd1)
        self.y_val21.append(self.x_state2)
        self.y_val22.append(self.gd2)
        self.y_val31.append(self.x_state3)
        self.y_val32.append(self.gd3)
        self.y_val41.append(self.x_state4)
        self.y_val42.append(self.gd4)
        self.y_val51.append(self.x_state5)
        self.y_val52.append(self.gd5)
        self.y_val61.append(self.x_state6)
        self.y_val62.append(self.gd6)
        """ Update x & y axis """
        xmin1, xmax1 = self.ax1.get_xlim()
        ymin1, ymax1 = self.ax1.get_ylim()
        xmin2, xmax2 = self.ax2.get_xlim()
        ymin2, ymax2 = self.ax2.get_ylim()
        xmin3, xmax3 = self.ax3.get_xlim()
        ymin3, ymax3 = self.ax3.get_ylim()
        xmin4, xmax4 = self.ax4.get_xlim()
        ymin4, ymax4 = self.ax4.get_ylim()
        xmin5, xmax5 = self.ax5.get_xlim()
        ymin5, ymax5 = self.ax5.get_ylim()
        xmin6, xmax6 = self.ax6.get_xlim()
        ymin6, ymax6 = self.ax6.get_ylim()
        if len(self.x_val) > xmax1:
            self.ax1.set_xlim(xmin1, 2 * xmax1)
            self.ax1.figure.canvas.draw()
            self.ax2.set_xlim(xmin1, 2 * xmax1)
            self.ax2.figure.canvas.draw()
            self.ax3.set_xlim(xmin1, 2 * xmax1)
            self.ax3.figure.canvas.draw()
            self.ax4.set_xlim(xmin1, 2 * xmax1)
            self.ax4.figure.canvas.draw()
            self.ax5.set_xlim(xmin1, 2 * xmax1)
            self.ax5.figure.canvas.draw()
            self.ax6.set_xlim(xmin1, 2 * xmax1)
            self.ax6.figure.canvas.draw()
        if min(self.x_state1, self.gd1) < ymin1:
            self.ax1.set_ylim(2 * ymin1, ymax1)
            self.ax6.figure.canvas.draw()
        elif max(self.x_state1, self.gd1) > ymax1:
            self.ax1.set_ylim(ymin1, 2 * ymax1)
            self.ax6.figure.canvas.draw()
        if min(self.x_state2, self.gd2) < ymin2:
            self.ax2.set_ylim(2 * ymin2, ymax2)
            self.ax6.figure.canvas.draw()
        elif max(self.x_state2, self.gd2) > ymax2:
            self.ax2.set_ylim(ymin2, 2 * ymax2)
            self.ax6.figure.canvas.draw()
        if min(self.x_state3, self.gd3) < ymin3:
            self.ax3.set_ylim(2 * ymin3, ymax3)
            self.ax6.figure.canvas.draw()
        elif max(self.x_state3, self.gd3) > ymax3:
            self.ax3.set_ylim(ymin3, 2 * ymax3)
            self.ax6.figure.canvas.draw()
        if min(self.x_state4, self.gd4) < ymin4:
            self.ax4.set_ylim(2 * ymin4, ymax4)
            self.ax6.figure.canvas.draw()
        elif max(self.x_state4, self.gd4) > ymax4:
            self.ax4.set_ylim(ymin4, 2 * ymax4)
            self.ax6.figure.canvas.draw()
        if min(self.x_state5, self.gd5) < ymin5:
            self.ax5.set_ylim(2 * ymin5, ymax5)
            self.ax6.figure.canvas.draw()
        elif max(self.x_state5, self.gd5) > ymax5:
            self.ax5.set_ylim(ymin5, 2 * ymax5)
            self.ax6.figure.canvas.draw()
        if min(self.x_state6, self.gd6) < ymin6:
            self.ax6.set_ylim(2 * ymin6, ymax6)
            self.ax6.figure.canvas.draw()
        elif max(self.x_state6, self.gd6) > ymax6:
            self.ax6.set_ylim(ymin6, 2 * ymax6)
            self.ax6.figure.canvas.draw()

        """ Update ln, """
        self.ln11.set_data(self.x_val, self.y_val11)
        self.ln12.set_data(self.x_val, self.y_val12)
        self.ln21.set_data(self.x_val, self.y_val21)
        self.ln22.set_data(self.x_val, self.y_val22)
        self.ln31.set_data(self.x_val, self.y_val31)
        self.ln32.set_data(self.x_val, self.y_val32)
        self.ln41.set_data(self.x_val, self.y_val41)
        self.ln42.set_data(self.x_val, self.y_val42)
        self.ln51.set_data(self.x_val, self.y_val51)
        self.ln52.set_data(self.x_val, self.y_val52)
        self.ln61.set_data(self.x_val, self.y_val61)
        self.ln62.set_data(self.x_val, self.y_val62)
        # Attention: Add commas to convert the returned params to tuples containing the params
        return self.ln11, self.ln12, self.ln21, self.ln22, self.ln31, self.ln32, self.ln41, self.ln42, self.ln51, self.ln52, self.ln61, self.ln62,
