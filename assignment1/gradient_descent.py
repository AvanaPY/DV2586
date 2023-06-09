import torch
import matplotlib.pyplot as plt

TORCH_DEVICE = torch.device('cpu')
STEP_SIZE = 1e-3
STEPS = 50_000
EARLY_STOPPING_THRESHOLD = 1e-4

def f(x : float, y : float):
    x = torch.tensor(x, dtype=torch.float32, device=TORCH_DEVICE, requires_grad=True)
    y = torch.tensor(y, dtype=torch.float32, device=TORCH_DEVICE, requires_grad=True)
    z = torch.square(x) + torch.square(y) + (x * (y + 2)) + torch.cos(3 * x)
    return (x, y), z

def gradient_descent(init_x : float, init_y : float):
    x = init_x
    y = init_y

    z_values = []
    d_values = []
    points = []
    for step in range(STEPS):
        (_x, _y), z = f(x, y)
        z.backward()            # Calculate derivatives
        
        dx = _x.grad            # Derivative with respect to x
        dy = _y.grad            # Derivative with respect to y
        total_d = (dx * dx + dy * dy) ** 0.5        # Estimate of total derivative, just the length of the derivative vector
        
        z_values.append(z.item())
        d_values.append((dx.item(), dy.item()))
        points.append((x, y))
        
        if step % 500 == 0:
            print(f'{step=}')
            print(f'    {z=}')
            print(f'    {x=}, {y=}')
            print(f'    {total_d=}')
        
        x -= dx.item() * STEP_SIZE     # Gradient descent
        y -= dy.item() * STEP_SIZE
            
        if total_d < EARLY_STOPPING_THRESHOLD:                          # Early stop if the derivative is small enough
            print(f'Early stopping at {step=}!')
            print(f'    {z=}')
            print(f'    {x=}, {y=}')
            print(f'    {total_d=}')
            break
    
    return z_values, d_values, points
    
def plot_gradient_descent(z_values, d_values, points):

    dx_values = [d[0] for d in d_values]
    dy_values = [d[1] for d in d_values]
    init_x, init_y = points[0]
    fin_x, fin_y = points[-1]

    xs = list(range(len(z_values)))

    fig = plt.figure(figsize=(10,10))
    fig.suptitle('Gradient Descent Results', fontsize=16)

    gs = fig.add_gridspec(2,2)

    ax = fig.add_subplot(gs[0, 0])
    ax.plot(xs, z_values, c='c')
    ax.set_xlabel("Step")
    ax.set_ylabel("Value")
    ax.set_title('Value of function')
    ax.legend(('f'))

    ax = fig.add_subplot(gs[0, 1])
    ax.plot(xs, dx_values, c='aquamarine')
    ax.plot(xs, dy_values, c='m')
    ax.set_xlabel("Step")
    ax.set_ylabel("Value")
    ax.set_title('Value derivatives')
    ax.legend(('dx', 'dy'))

    scatter_x = [point[0] for point in points]
    scatter_y = [point[1] for point in points]

    ax = fig.add_subplot(gs[1, 0])
    ax.scatter(scatter_x[:1], scatter_y[:1], c='r')
    ax.scatter(scatter_x[-1:], scatter_y[-1:], c='g')
    ax.plot(scatter_x, scatter_y, markersize=1, marker='1')
    ax.set_ylabel('y')
    ax.set_xlabel('x')
    ax.set_title('Plot of (x,y) movement')
    ax.grid(True)
    ax.legend((f'Start ({init_x:.2f}, {init_y:.2f})', f'End ({fin_x:.2f}, {fin_y:.2f})', '(x,y) movement'))

    ax = fig.add_subplot(gs[1, 1])
    ax.plot(xs, scatter_x)
    ax.plot(xs, scatter_y)
    ax.legend(('x', 'y'))
    ax.set_xlabel('Step')
    ax.set_ylabel('Value')
    ax.set_title('Plot of X and Y values over time')
    plt.show()

z_values, d_values, points = gradient_descent(init_x=-3.2, init_y=7.8)
plot_gradient_descent(z_values, d_values, points)