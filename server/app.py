from flask import Flask, request, jsonify
import matplotlib.pyplot as plt, mpld3
import content.parabolic as parabolic
import numpy as np

def web_params_parse(args):
    prefix = 'param' + '-'
    params = {}
    for name, value in args.items():
        if name.startswith(prefix):
            params[name[len(prefix):].replace('-', '_')] = value
    return params


app = Flask(__name__)

def get_errors(U, reference):
    res = np.zeros((len(U)))
    for i in range(len(U)):
        for j in range(len(U[i])):
            res[i] += abs(reference[i, j] - U[i, j])
        res[i] /= len(U[i])
    return res


@app.route('/parabolic/solve', methods=['POST'])
def parabolic_equation_solve():
    params = web_params_parse(request.get_json())

    solve = parabolic.Solver(params)

    fig=plt.figure(figsize=(8, 8))
    solve_results_dict = solve.getResultsDict()
    exact_solution = solve_results_dict['exact_solution']
    calculated_solution = solve_results_dict['U']
    vec_t = solve_results_dict['t']

    import pprint
    pprint.pprint(exact_solution)
    pprint.pprint(calculated_solution)
    fig.add_subplot(222).set_title("Точное решение")
    plt.imshow(exact_solution)
    
    fig.add_subplot(221).set_title("Решение по выбранной схеме")
    plt.imshow(calculated_solution)

    fig.add_subplot(212).set_title("График ошибок")
    errors = get_errors(calculated_solution, exact_solution)
    print(exact_solution, vec_t, errors)
    plt.plot(vec_t, errors)

    html = mpld3.fig_to_html(fig)
    return jsonify(html=html)
