from app import app
import numpy as np
import dash_core_components as dcc
import dash_html_components as html
from plotly import graph_objs as go
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate


def generate_input_label(label, id, value, type):
    return html.Div(
        className="div-for-dropdown",
        children=[
            html.Label(label),
            dcc.Input(
                id=id,
                value=value,
                type=type,
            ),
        ])


def generate_controls():
    return [
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

        html.Button(id='run-algo', children='Run the algo !'),
        html.Button(id='show-code', children='Show me the code !')
    ]


def generate_graphs():
    return [
        html.Div(
            className="columns",
            children=[
                dcc.Graph(id="heatmap-initial")
            ]
        ),
        html.Div(
            className="six columns",
            children=[
                dcc.Graph(id="heatmap-completion")
            ]
        )
    ]


def generate_layout_4controls_8graphs(controls, graphs):
    return html.Div([
        # The whole app is a row
        html.Div(
            className="row",
            children=[
                # Column for user controls
                html.Div(
                    className="four columns div-user-controls",
                    children=controls
                ),

                # Column for app graphs and plots
                html.Div(
                    className="eight columns",
                    children=graphs
                ),
            ])
    ])


controls = generate_controls()
graphs = generate_graphs()
layout = generate_layout_4controls_8graphs(controls, graphs)


@app.callback(
    [Output('heatmap-initial', 'figure'),
     Output('heatmap-completion', 'figure')],
    [Input('run-algo', 'n_clicks'),
     Input('num-users', 'value'),
     Input('num-evals', 'value')])
def update_heatmap_completion(n_clicks, num_users, num_evals):
    #if n_clicks is None:
    #    raise PreventUpdate

    solver = MatrixCompletion(num_users, num_evals)
    print('Running...')
    solver.solve()
    print('Done !')

    layout = go.Layout(
        bargap=0.01,
        bargroupgap=0,
        barmode="group",
        margin=go.layout.Margin(l=10, r=0, t=0, b=50),
        showlegend=False,
        plot_bgcolor="#323130",
        paper_bgcolor="#323130",
        dragmode="select",
        font=dict(color="white"),
        xaxis=dict(
            range=[-0.5, num_evals],
            showgrid=False,
            nticks=num_evals,
            fixedrange=True,
        ),
        yaxis=dict(
            range=[0, num_evals],
            showticklabels=False,
            showgrid=False,
            fixedrange=True,
            rangemode="nonnegative",
            zeroline=False,
        )
    )

    return [
        go.Figure(
            data=[
                go.Heatmap(
                    z=solver.A,
                    x=np.arange(num_evals),
                    y=np.arange(num_users)
                )
            ],
            layout=layout
        ),
        go.Figure(
            data=[
                go.Heatmap(
                    z=solver.X * solver.M,
                    x=np.arange(num_evals),
                    y=np.arange(num_users)
                )
            ],
            layout=layout
        )
    ]


class MatrixCompletion:
    """
    Soit A une matrice clairsemée de dimensions (p, q).
    On suppose que A soit de rang faible.

    Soit M une matrice de dimensions (p, q) dite masque.
    M(i,j) = 1 ssi A(i, j) est défini, 0 sinon.
    """
    def __init__(self, rows, cols):
        self.rows, self.cols = rows, cols

        self.M = self.generate_mask(rows, cols)
        self.A = self.generate_A(rows, cols)
        self.X = self.generate_X(rows, cols)

    def generate_mask(self, rows, cols):
        self.M = np.random.choice([0, 1], (rows, cols), p=[0.7, 0.3])
        return self.M

    def generate_X(self, rows, cols):
        self.X = np.random.randint(0, 5, (rows, cols))
        return self.X

    def generate_A(self, rows, cols):
        # A_temp = np.random.random((rows, cols))
        # A_temp = A_temp @ A_temp.T
        self.A = np.random.randint(0, 5, (rows, cols)) * self.M
        # self.A = A_temp * self.M
        return self.A

    def compute_solution(self, x_sol):
        return 1/2 * np.linalg.norm(self.M * (self.A - x_sol))

    def gradient_descent(self, max_iterations, gradient_step, epsilon):
        x_t = self.X
        differences = []

        for it in range(max_iterations):
            x_temp = x_t + gradient_step * (self.M * (self.A - x_t))

            # décomposition SVD
            evaluations, diagonal, users = np.linalg.svd(x_temp)

            # les valeurs singulières sont triées par ordre décroissant
            diagonal_opt = np.diag(diagonal[:self.cols])
            diagonal_opt.resize(self.A.shape)

            x_t = evaluations @ diagonal_opt @ users

            current_distance = self.compute_solution(x_temp)
            differences.append(current_distance)

            if current_distance <= epsilon:
                print('Current distance {}'.format(current_distance))
                print('Converged earlier at iteration {}'.format(it))
                return x_t

        return x_t

    def solve(self, max_iterations=100_000, gradient_step=0.001, epsilon=0.01):
        self.X = self.gradient_descent(max_iterations, gradient_step, epsilon)
