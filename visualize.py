#!/usr/bin/env python
from keras.models import model_from_json
from IPython.display import SVG
from keras.utils.visualize_util import plot, model_to_dot


def main():
    with open('data/cafa3/model_cc.json') as f:
        json_string = next(f)
    model = model_from_json(json_string)

    graph = model_to_dot(model).create(prog='dot', format='svg')
    with open('graph.svg', 'w') as f:
        f.write(SVG(graph).data)


if __name__ == '__main__':
    main()
