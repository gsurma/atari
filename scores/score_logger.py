from statistics import mean
import matplotlib.pyplot as plt
from collections import deque
import os
import csv
import numpy as np


class ScoreLogger:

    def __init__(self, header, path):
        self.scores = deque()
        self.steps = deque()
        self.path = path
        self.header = header
        if not os.path.exists(self.path):
            os.makedirs(self.path)

    def add_score(self, score):
        self._save_csv(self.path + "scores.csv", score)
        self._save_png(input_path=self.path + "scores.csv",
                       output_path=self.path + "scores.png",
                       x_label="runs",
                       y_label="scores",
                       average_of_n_last=100)
        self.scores.append(score)
        mean_score = mean(self.scores)
        print "Scores: (min: " + str(min(self.scores)) + ", avg: " + str(mean_score) + ", max: " + str(max(self.scores))

    def add_run_duration(self, step):
        self._save_csv(self.path + "steps.csv", step)
        self._save_png(input_path=self.path + "steps.csv",
                       output_path=self.path + "steps.png",
                       x_label="runs",
                       y_label="steps",
                       average_of_n_last=100)
        self.steps.append(step)
        mean_step = mean(self.steps)
        print "Steps: (min: " + str(min(self.steps)) + ", avg: " + str(mean_step) + ", max: " + str(max(self.steps))

    def _save_png(self, input_path, output_path, x_label, y_label, average_of_n_last):
        x = []
        y = []
        with open(input_path, "r") as scores:
            reader = csv.reader(scores)
            data = list(reader)
            for i in range(0, len(data)):
                x.append(int(i))
                y.append(int(data[i][0]))

        plt.subplots()
        plt.plot(x, y, label="score per run")

        average_range = average_of_n_last if average_of_n_last is not None else len(x)
        plt.plot(x[-average_range:], [np.mean(y[-average_range:])] * len(y[-average_range:]), linestyle="--", label="last " + str(average_range) + " average")

        if len(x) > 1:
            trend_x = x[1:]
            z = np.polyfit(np.array(trend_x), np.array(y[1:]), 1)
            p = np.poly1d(z)
            plt.plot(trend_x, p(trend_x), linestyle="-.",  label="trend")

        plt.title(self.header)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.legend(loc="upper left")
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()

    def _save_csv(self, path, score):
        if not os.path.exists(path):
            with open(path, "w"):
                pass
        scores_file = open(path, "a")
        with scores_file:
            writer = csv.writer(scores_file)
            writer.writerow([score])
