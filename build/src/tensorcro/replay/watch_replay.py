# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
#   This file was created by: Alberto Palomo Alonso         #
# Universidad de Alcalá - Escuela Politécnica Superior      #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
import matplotlib
import json
import os
import tkinter as tk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        FUNCTION DEF                       #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
def watch_replay(path) -> None:
    root = tk.Tk()
    ReplayGUI(root, path)
    root.mainloop()


# Tkinter GUI
class ReplayGUI:
    def __init__(self, master, path):
        self.master = master
        self.len_path = len(os.listdir(path)) // 2
        self.fitness, self.reef, self.names = self.load_all(path)
        self.index = 0
        self.master.title(f"Coral Reef Optimization Replay")
        self.master.geometry("955x710")
        self.label = tk.Label(master, text=f"Generation 0 / {self.len_path}")
        self.label.pack()

        self.next_reef = tk.Button(master, text="------>", command=self.next)
        self.next_reef.place(x=550, y=15)

        self.last_reef = tk.Button(master, text="<------", command=self.last)
        self.last_reef.place(x=10, y=15)

        self.img_lf = tk.LabelFrame(master, width=370, height=370)
        self.img_lf.place(x=10, y=55)

        self.table = tk.Text(master, width=40, height=38, bd=2, wrap='none')
        self.table.place(x=620, y=50)

        self.top_label_fitness = tk.Label(master, text="Best Corals", font=("Arial", 20))
        self.top_label_fitness.place(x=710, y=10)

        self.static_view = tk.Button(master, text="<>", command=self.static)
        self.static_view.place(x=80, y=15)

        self.static_view2 = tk.Button(master, text="◇", command=self.static2)
        self.static_view2.place(x=510, y=15)

        self.canvas1 = None
        self.toolbar1 = None
        self.ax = None
        self.render()
        self.list_fitness()

    def next(self):
        self.index += 1
        if self.index >= len(self.fitness):
            self.index = 0
        self.render()
        self.list_fitness()

    def last(self):
        self.index -= 1
        if self.index < 0:
            self.index = len(self.fitness) - 1
        self.render()
        self.list_fitness()

    def render(self):
        reef = self.fitness[self.index]
        names = self.names['names']
        shards = self.names['shards']
        self.label.destroy()
        self.label = tk.Label(self.master, text=f"Epoch {shards * (self.index + 1)} / {shards * len(self.fitness)}",
                              font=("Arial", 20))
        self.label.place(x=240, y=10)
        padding = reef.shape[1] // len(names)
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection='3d')
        __one_line_colors = np.array([[np.array(cm.tab10(int(_ // padding)))] for _ in range(reef.shape[1])])
        kol = np.reshape(np.tile(__one_line_colors, [reef.shape[0], 1]), (reef.shape[0], reef.shape[1], 4))
        _x = np.arange(reef.shape[0])
        _y = np.arange(reef.shape[1])
        _xx, _yy = np.meshgrid(_x, _y)
        x, y = _xx.ravel(), _yy.ravel()
        top = np.where(reef < 0, 0, np.exp(reef)).ravel()
        bottom = np.zeros_like(top)
        ax.bar3d(x, y, bottom, 2, 1, top, shade=True, alpha=0.5, linewidth=3., linestyle='-',
                 color=np.reshape(kol, (-1, 4)))
        ax.set_zticks([])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.w_zaxis.line.set_lw(0.)
        ax.w_xaxis.line.set_c('w')
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((0.0, 0.0, 0.0, 1.0))
        ax.set_title(f'CRO Fitness epoch {shards * self.index}')
        ax.grid(False)
        handles = [matplotlib.patches.Patch(color=cm.tab10(_), label=name) for _, name in enumerate(names)]
        ax.legend(handles=handles, fontsize='8', ncol=3)
        if self.ax is not None:
            ax.view_init(elev=self.ax.elev, azim=self.ax.azim)
        self.ax = ax
        if self.canvas1 is not None:
            self.canvas1.get_tk_widget().pack_forget()
            self.toolbar1.destroy()
        self.canvas1 = FigureCanvasTkAgg(fig, master=self.img_lf)
        self.canvas1.draw()
        self.toolbar1 = NavigationToolbar2Tk(self.canvas1, self.img_lf)
        self.toolbar1.update()
        self.canvas1.get_tk_widget().pack()
        plt.close(fig)

    def list_fitness(self):
        flat_reef = self.reef[self.index].reshape(-1, self.reef[self.index].shape[-1])
        flat_fitness = self.fitness[self.index].flatten()
        # Sort by fitness
        sorted_fitness = np.argsort(flat_fitness)[::-1]
        sorted_reef = flat_reef[sorted_fitness]
        sorted_fitness = flat_fitness[sorted_fitness]
        df = pd.DataFrame(sorted_reef, columns=[f'Param. {_}' for _ in range(flat_reef.shape[-1])])
        df['Fitness'] = sorted_fitness
        self.table.delete('1.0', tk.END)
        # Put fitness as first column:
        cols = df.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        df = df[cols]
        self.table.insert(tk.INSERT, df.to_string(formatters={key: lambda x: f'{x:.4f}' for key in df.columns}))

    def static(self):
        if self.ax.elev != 0 or self.ax.azim != 0:
            self.ax.view_init(elev=0, azim=0)
            self.render()

    def static2(self):
        if self.ax.elev != 90 or self.ax.azim != 0:
            self.ax.view_init(elev=90, azim=0)
            self.render()

    def load_all(self, path):
        fitness = list()
        reef = list()
        for i, _path in enumerate(os.listdir(path)):
            if i >= self.len_path:
                break
            else:
                fitness.append(np.load(path + f'/fitness_{i + 1}.npy'))
                reef.append(np.load(path + f'/reef_{i + 1}.npy'))
        with open(f'{path}/config.json', 'rb') as f:
            names = json.load(f)
        return fitness, reef, names
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
