import numpy as np
import dash_html_components as html
import dash_core_components as dcc


def generate_input_label(label, id, value, type):
    return html.Div([
        html.Label(label),
        dcc.Input(
            id=id,
            value=value,
            type=type,
        ),
    ])


layout = html.Div([
    html.Div([
        generate_input_label('Number of users',
                             'num-users',
                             15, 'number'),
        generate_input_label('Number of evals',
                             'num-evals',
                             3, 'number'),
        generate_input_label('Gradient descent max iterations',
                             'max-iters',
                             100_000, 'number'),
        generate_input_label('Gradient descent step',
                             'gradient-step',
                             0.001, 'number'),
        generate_input_label('Gradient descent convergence',
                             'epsilon',
                             0.001, 'number'),

        html.Button(id='run-algo', children='Run the algo !')],
        style={'width': '49%', 'display': 'inline-block'}),

    html.Div(id=''),

    dcc.Graph(id='heatmap-initial'),
    dcc.Graph(id='heatmap-completion')
], style={
    'className': 'container'
})


class LagrangianSolver:
    def __init__(self, dim):
        self.A = self.generate_a(dim, mu=0, sigma=0.1)
        self.x = self.generate_x(dim[1], distribution=[0.8, 0.2])
        self.nu = self.generate_nu(dim[1])

        # Ax = b
        self.b = self.A @ self.x

        # transposes
        self.A_T = np.transpose(self.A)
        self.b_T = np.transpose(self.b)
        self.nu_T = np.transpose(self.nu)

    def generate_a(self, dim, mu, sigma):
        self.A = np.random.normal(mu, sigma, dim)
        return self.A

    def generate_x(self, cols, distribution):
        self.x = np.random.choice([0, 1], size=(cols, ), p=distribution)
        return self.x

    def generate_nu(self, cols):
        self.nu = np.random.rand(cols)
        return self.nu

    def gradient_ascent(self, nu_analytic, iterations, step, do_plot):
        # TODO : stop ascent if epsilon condition is met
        nu_opt = nu_analytic

        saved_gradients = list()
        saved_l2_norms = list()

        for i in range(iterations):
            if do_plot:
                saved_gradients.append(self.compute_x(nu_opt))
                saved_l2_norms.append(np.linalg.norm(self.x -
                                                     self.compute_x(nu_opt)))
            nu_opt = nu_opt - step * (1/2 * self.A @ self.A.T @ nu_opt + self.b)

        """
        if do_plot:
            plt.plot(range(iterations), saved_gradients)
            plt.plot(range(iterations), saved_l2_norms,
                     color='Black', label='Euclidian distance')
            plt.xlabel('Iterations')
            plt.legend('upper left')
            plt.show()
        """

        return nu_opt

    def compute_x(self, nu_opt):
        return -1/2 * self.A.T @ nu_opt

    def solve_dual(self, iterations=100_000, step=0.01, do_plot=True):
        # we resolve a new expression by derivation of nu in norm 2
        print(self.nu.T.shape)
        print(self.A.shape)
        print(self.A.T.shape)
        print(self.nu.shape)

        starting_block = np.zeros(self.A.shape[0])
        # starting_block = np.zeros(self.b)

        # using gradient ascent to optimize nu
        nu_opt = self.gradient_ascent(starting_block, iterations, step, do_plot)

        # sol_analytic = -1/4 * nu_opt.T @ self.A @ self.A.T @ nu_opt - self.b.T @ nu

        # back to the initial equation
        return self.compute_x(nu_opt)

    def solve_lagrange(self):
        pass


if __name__ == "__main__":
    solver = LagrangianSolver((24, 20))

    # dual solver
    dual_sol = solver.solve_dual()

    print('Initial vector : ', solver.x)
    print('Vector found : ', np.around(dual_sol))

    print('L2 norm : ', np.linalg.norm(solver.x - dual_sol))
