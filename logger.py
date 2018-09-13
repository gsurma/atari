from statistics import mean
import matplotlib.pyplot as plt
from collections import deque
import os
import csv
import numpy as np


class Logger:

    def __init__(self, header, path):
        self.scores = deque()
        self.steps = deque()
        self.losses = deque()
        self.accuracies = deque()
        self.path = path
        self.header = header
        # TODO: add avg human scores
        if not os.path.exists(self.path):
            os.makedirs(self.path)

    def add_accuracy(self, accuracy):
        self._save_csv(self.path + "accuracies.csv", accuracy)
        self._save_png(input_path=self.path + "accuracies.csv",
                       output_path=self.path + "accuracies.png",
                       x_label="updates",
                       y_label="accuracies",
                       xy_label="average accuracy per update",
                       batch_average_length=1000)
        self.accuracies.append(accuracy)
        mean_accuracy = mean(self.accuracies)
        print "Accuracies: (min: " + str(min(self.accuracies)) + ", avg: " + str(mean_accuracy) + ", max: " + str(max(self.accuracies))

    def add_loss(self, loss):
        self._save_csv(self.path + "losses.csv", loss)
        self._save_png(input_path=self.path + "losses.csv",
                       output_path=self.path + "losses.png",
                       x_label="updates",
                       y_label="losses",
                       xy_label="average loss per update",
                       batch_average_length=1000)
        self.losses.append(loss)
        mean_loss = mean(self.losses)
        print "Losses: (min: " + str(min(self.losses)) + ", avg: " + str(mean_loss) + ", max: " + str(max(self.losses))

    def add_score(self, score):
        self._save_csv(self.path + "scores.csv", score)
        self._save_png(input_path=self.path + "scores.csv",
                       output_path=self.path + "scores.png",
                       x_label="runs",
                       y_label="scores",
                       xy_label="score per run",
                       batch_average_length=100)
        self.scores.append(score)
        mean_score = mean(self.scores)
        print "Scores: (min: " + str(min(self.scores)) + ", avg: " + str(mean_score) + ", max: " + str(max(self.scores))

    def add_run_duration(self, step):
        self._save_csv(self.path + "steps.csv", step)
        self._save_png(input_path=self.path + "steps.csv",
                       output_path=self.path + "steps.png",
                       x_label="runs",
                       y_label="steps",
                       xy_label="steps per run",
                       batch_average_length=100)
        self.steps.append(step)
        mean_step = mean(self.steps)
        print "Steps: (min: " + str(min(self.steps)) + ", avg: " + str(mean_step) + ", max: " + str(max(self.steps))

    def _save_png(self, input_path, output_path, x_label, y_label, xy_label, batch_average_length):
        x = []
        y = []
        with open(input_path, "r") as scores:
            reader = csv.reader(scores)
            data = list(reader)
            for i in range(0, len(data)):
                x.append(float(i))
                y.append(float(data[i][0]))

        plt.subplots()
        plt.plot(x, y, label=xy_label)

        batch_averages_y = []
        batch_averages_x = []
        temp_values_in_batch = []
        for i in xrange(len(y)):
            temp_values_in_batch.append(y[i])
            if i % batch_average_length == 0 and i != 0:
                batch_averages_y.append(mean(temp_values_in_batch))
                batch_averages_x.append(len(batch_averages_y)*batch_average_length-batch_average_length)
                temp_values_in_batch = []
        if batch_averages_x and batch_averages_y:
            plt.plot(batch_averages_x, batch_averages_y, linestyle="--", label="last " + str(batch_average_length) + " average")

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
