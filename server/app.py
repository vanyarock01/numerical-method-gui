from flask import (
    Flask,
    request,
    jsonify
)
from content import (
    parabolic,
    hyperbolic,
    elliptic
)

import matplotlib.pyplot as plt
import mpld3
import numpy as np


def web_params_parse(args, lab_num):
    prefix = 'param-{0}-'.format(str(lab_num))
    params = {}
    for name, value in args.items():
        if name.startswith(prefix):
            params[name[len(prefix):].replace('-', '_')] = value
    return params


def get_solve_errors(U, reference):
    res = np.zeros((len(U)))
    for i in range(len(U)):
        for j in range(len(U[i])):
            res[i] += abs(reference[i, j] - U[i, j])
        res[i] /= len(U[i])
    return res


app = Flask(__name__)

@app.route('/parabolic/solve', methods=['POST'])
def parabolic_equation_solve():
    params = web_params_parse(request.get_json(), 1)

    solve = parabolic.Solver(params)
    solve_results_dict = solve.getResultsDict()

    fig = plt.figure(figsize=(8, 8))

    exact_solution = solve_results_dict['exact_solution']
    fig.add_subplot(222).set_title("Точное решение")
    plt.imshow(exact_solution)

    calculated_solution = solve_results_dict['U']
    fig.add_subplot(221).set_title("Решение по выбранной схеме")
    plt.imshow(calculated_solution)

    fig.add_subplot(212).set_title("График ошибок")
    plt.plot(solve_results_dict['t'], get_solve_errors(calculated_solution, exact_solution))

    return jsonify(html=mpld3.fig_to_html(fig))


@app.route('/hyperbolic/solve', methods=['POST'])
def hyperbolic_equation_solve():
    params = web_params_parse(request.get_json(), 2)

    solve = hyperbolic.Solver(params)
    solve_results_dict = solve.getResultsDict()

    fig = plt.figure(figsize=(8, 8))

    exact_solution = solve_results_dict['exact_solution']
    fig.add_subplot(222).set_title("Точное решение")
    plt.imshow(exact_solution)

    calculated_solution = solve_results_dict['U']
    fig.add_subplot(221).set_title("Решение по выбранной схеме")
    plt.imshow(calculated_solution)

    fig.add_subplot(212).set_title("График ошибок")
    plt.plot(solve_results_dict['t'], get_solve_errors(calculated_solution, exact_solution))

    return jsonify(html=mpld3.fig_to_html(fig))


@app.route('/elliptic/solve', methods=['POST'])
def elliptic_equation_solve():
    params = web_params_parse(request.get_json(), 3)

    solve = elliptic.Solver(params)
    solve_results_dict = solve.getResultsDict()

    fig = plt.figure(figsize=(8, 8))

    exact_solution = solve_results_dict['exact_solution']
    fig.add_subplot(222).set_title("Точное решение")
    plt.imshow(exact_solution)

    calculated_solution = solve_results_dict['U']
    fig.add_subplot(221).set_title("Решение по выбранной схеме")
    plt.imshow(calculated_solution)

    fig.add_subplot(212).set_title("График ошибок")
    plt.plot(get_solve_errors(calculated_solution, exact_solution))
    from pprint import pprint
    pprint(solve_results_dict['iterations'])
    pprint(get_solve_errors(calculated_solution, exact_solution))
    # plt.plot(solve_results_dict['iterations'], get_solve_errors(calculated_solution, exact_solution))

    return jsonify(html=mpld3.fig_to_html(fig))