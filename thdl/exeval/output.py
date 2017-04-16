# -*- coding: utf-8 -*-


def print_runout_history_metric(history_aspect_all_output_lines, file):
    maxlen = max([len(line) for output_lines in history_aspect_all_output_lines for line in output_lines]) + 1

    for lines in zip(*history_aspect_all_output_lines):
        runout = ''
        for line in lines:
            runout += (line + " " * (maxlen - len(line)))
        print(runout, file=file)



