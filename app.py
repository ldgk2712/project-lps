from flask import Flask, render_template, request, send_file
import io
import base64
from simplex import solve_simplex
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Đặt backend Agg

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    default_num_vars = 2
    default_num_constraints = 2

    if request.method == 'POST':
        try:
            num_vars_input = request.form.get('num_vars', default_num_vars)
            num_constraints_input = request.form.get('num_constraints', default_num_constraints)

            # validate inputs
            if not num_vars_input.isdigit() or int(num_vars_input) < 1:
                return render_template('index.html', num_vars=num_vars_input,
                                       num_constraints=num_constraints_input,
                                       error="Số lượng biến (n) phải là số nguyên dương (≥1).")
            if not num_constraints_input.isdigit() or int(num_constraints_input) < 1:
                return render_template('index.html', num_vars=num_vars_input,
                                       num_constraints=num_constraints_input,
                                       error="Số lượng ràng buộc (m) phải là số nguyên dương (≥1).")

            num_vars = int(num_vars_input)
            num_constraints = int(num_constraints_input)
            if 'update' in request.form:
                return render_template('index.html', num_vars=num_vars,
                                       num_constraints=num_constraints)

            # read parameters
            c = [float(request.form[f'c{i}']) for i in range(num_vars)]
            A, b, constraint_types = [], [], []
            for i in range(num_constraints):
                A.append([float(request.form[f'A{i}_{j}']) for j in range(num_vars)])
                b.append(float(request.form[f'b{i}']))
                constraint_types.append(request.form[f'constraint_type{i}'])
            variable_types = [request.form[f'var_type{i}'] for i in range(num_vars)]

            # solve
            result = solve_simplex(A, b, c, constraint_types,
                                   request.form['objective_type'], variable_types)
            if result['status'].startswith('Lỗi'):
                return render_template('index.html', num_vars=num_vars,
                                       num_constraints=num_constraints,
                                       error=result['status'])

            # plot if 2 variables
            plot_data = None
            if num_vars == 2 and result['status'] == 'Tối ưu (Optimal)':
                plot_data = create_plot(A, b, constraint_types,
                                        c, result['solution'], variable_types)

            return render_template('result.html', result=result, plot_data=plot_data)

        except (ValueError, KeyError):
            return render_template('index.html', num_vars=num_vars_input,
                                   num_constraints=num_constraints_input,
                                   error="Vui lòng nhập đầy đủ và đúng định dạng số.")

    return render_template('index.html', num_vars=default_num_vars,
                           num_constraints=default_num_constraints)


def create_plot(A, b, constraint_types, c, solution, variable_types):
    fig, ax = plt.subplots(figsize=(8, 8))

    # Compute all candidate vertices (pairwise intersections)
    vertices = []
    n = len(A)
    # Intersections of linear constraints
    for i in range(n):
        for j in range(i + 1, n):
            a1, a2 = A[i]
            c1, c2 = A[j]
            det = a1 * c2 - a2 * c1
            if abs(det) < 1e-9:
                continue
            x = (b[i] * c2 - b[j] * a2) / det
            y = (a1 * b[j] - b[i] * c1) / det
            # Check feasibility
            ok = True
            for k in range(n):
                ak1, ak2 = A[k]
                val = ak1 * x + ak2 * y
                tp = constraint_types[k]
                if tp == '<=' and val > b[k] + 1e-6: ok = False
                if tp == '>=' and val < b[k] - 1e-6: ok = False
                if tp == '=' and abs(val - b[k]) > 1e-6: ok = False
                if not ok: break
            # Check variable bounds
            if ok:
                for idx, vt in enumerate(variable_types):
                    val = [x, y][idx]
                    if vt == '>=0' and val < -1e-6: ok = False
                    if vt == '<=0' and val > 1e-6: ok = False
                if ok:
                    vertices.append((x, y))
    
    # Intersections with variable bounds (x1=0, x2=0)
    for i in range(n):
        a1, a2 = A[i]
        b_val = b[i]
        # With x1=0
        if abs(a2) > 1e-9:
            y = b_val / a2
            x = 0
            ok = True
            for k in range(n):
                val = A[k][0] * x + A[k][1] * y
                tp = constraint_types[k]
                if tp == '<=' and val > b[k] + 1e-6: ok = False
                if tp == '>=' and val < b[k] - 1e-6: ok = False
                if tp == '=' and abs(val - b[k]) > 1e-6: ok = False
                if not ok: break
            if ok:
                for idx, vt in enumerate(variable_types):
                    val = [x, y][idx]
                    if idx == 0 and vt == '<=0' and val > 1e-6: ok = False
                    elif idx == 1:
                        if vt == '>=0' and val < -1e-6: ok = False
                        if vt == '<=0' and val > 1e-6: ok = False
                if ok:
                    vertices.append((x, y))
        # With x2=0
        if abs(a1) > 1e-9:
            x = b_val / a1
            y = 0
            ok = True
            for k in range(n):
                val = A[k][0] * x + A[k][1] * y
                tp = constraint_types[k]
                if tp == '<=' and val > b[k] + 1e-6: ok = False
                if tp == '>=' and val < b[k] - 1e-6: ok = False
                if tp == '=' and abs(val - b[k]) > 1e-6: ok = False
                if not ok: break
            if ok:
                for idx, vt in enumerate(variable_types):
                    val = [x, y][idx]
                    if idx == 1 and vt == '<=0' and val > 1e-6: ok = False
                    elif idx == 0:
                        if vt == '>=0' and val < -1e-6: ok = False
                        if vt == '<=0' and val > 1e-6: ok = False
                if ok:
                    vertices.append((x, y))
    
    # Include origin and optimal
    vertices.append((0, 0))
    if solution:
        vertices.append((solution['x1'], solution['x2']))

    # Determine plot bounds
    xs, ys = zip(*vertices)
    margin = 3.0
    x_min, x_max = min(xs) - margin, max(xs) + margin
    y_min, y_max = min(ys) - margin, max(ys) + margin

    # Meshgrid
    X, Y = np.meshgrid(np.linspace(x_min, x_max, 400),
                       np.linspace(y_min, y_max, 400))

    # Build mask
    mask = np.ones_like(X, dtype=bool)
    # Linear constraints
    for k, (ak1, ak2) in enumerate(A):
        if constraint_types[k] == '<=':
            mask &= (ak1 * X + ak2 * Y <= b[k] + 1e-6)
        elif constraint_types[k] == '>=':
            mask &= (ak1 * X + ak2 * Y >= b[k] - 1e-6)
        else:
            mask &= (np.abs(ak1 * X + ak2 * Y - b[k]) < 1e-2)
    # Variable bounds
    for idx, vt in enumerate(variable_types):
        if vt == '>=0':
            if idx == 0: mask &= (X >= -1e-6)
            else: mask &= (Y >= -1e-6)
        elif vt == '<=0':
            if idx == 0: mask &= (X <= 1e-6)
            else: mask &= (Y <= 1e-6)
        # URS: no additional constraint

    # Plot feasible region
    ax.contourf(X, Y, mask, levels=[0.5, 1], colors=['#cccccc'], alpha=0.6)

    # Plot constraint lines
    x_line = np.linspace(x_min, x_max, 400)
    for k, (ak1, ak2) in enumerate(A):
        label = f'Ràng buộc {k+1}'
        if abs(ak2) > 1e-9:
            y_line = (b[k] - ak1 * x_line) / ak2
            ax.plot(x_line, y_line, linewidth=2, label=label)
        elif abs(ak1) > 1e-9:
            ax.axvline(b[k] / ak1, linewidth=2, label=label)
    
    # Plot variable boundaries
    for idx, vt in enumerate(variable_types):
        if vt == '>=0':
            if idx == 0: ax.axvline(0, color='k', linestyle='-', linewidth=1, label='x1 ≥ 0')
            else: ax.axhline(0, color='k', linestyle='-', linewidth=1, label='x2 ≥ 0')
        elif vt == '<=0':
            if idx == 0: ax.axvline(0, color='k', linestyle='-', linewidth=1, label='x1 ≤ 0')
            else: ax.axhline(0, color='k', linestyle='-', linewidth=1, label='x2 ≤ 0')

    # Draw arrows for linear constraints
    for k, (ak1, ak2) in enumerate(A):
        if abs(ak2) > 1e-9:
            y_vals = (b[k] - ak1 * x_line) / ak2
            valid_idx = np.where((y_vals >= y_min) & (y_vals <= y_max))[0]
            if valid_idx.size > 0:
                idx0 = valid_idx[len(valid_idx) // 2]
                x0, y0 = x_line[idx0], y_vals[idx0]
            else:
                x0 = (x_min + x_max) / 2
                y0 = (b[k] - ak1 * x0) / ak2
        else:
            x0 = b[k] / ak1
            y_vals = np.linspace(y_min, y_max, 400)
            y0 = y_vals[len(y_vals) // 2]
        norm = np.hypot(ak1, ak2)
        dx, dy = -ak1 / norm, -ak2 / norm
        if constraint_types[k] == '>=':
            dx, dy = -dx, -dy
        ax.quiver(x0, y0, dx, dy, angles='xy', scale_units='xy', scale=4, color='blue', headlength=7, headwidth=4)
        ax.text(x0 + 0.1, y0 + 0.1, f'({k+1})', color='blue')
    
    # Arrows for variable bounds
    for idx, vt in enumerate(variable_types):
        if vt == '>=0':
            if idx == 0:
                ax.quiver(0, (y_min + y_max) / 2, 1, 0, angles='xy', scale_units='xy', scale=4, color='blue', headlength=7, headwidth=4)
                ax.text(0.1, (y_min + y_max) / 2 + 0.1, f'(x{idx+1} ≥ 0)', color='blue')
            else:
                ax.quiver((x_min + x_max) / 2, 0, 0, 1, angles='xy', scale_units='xy', scale=4, color='blue', headlength=7, headwidth=4)
                ax.text((x_min + x_max) / 2 + 0.1, 0.1, f'(x{idx+1} ≥ 0)', color='blue')
        elif vt == '<=0':
            if idx == 0:
                ax.quiver(0, (y_min + y_max) / 2, -1, 0, angles='xy', scale_units='xy', scale=4, color='blue', headlength=7, headwidth=4)
                ax.text(0.1, (y_min + y_max) / 2 + 0.1, f'(x{idx+1} ≤ 0)', color='blue')
            else:
                ax.quiver((x_min + x_max) / 2, 0, 0, -1, angles='xy', scale_units='xy', scale=4, color='blue', headlength=7, headwidth=4)
                ax.text((x_min + x_max) / 2 + 0.1, 0.1, f'(x{idx+1} ≤ 0)', color='blue')

    # Objective line at optimum
    # if 'objectiveValue' in solution:
    #     Z = solution['objectiveValue']
    #     if abs(c[1]) > 1e-9:
    #         ax.plot(x_line, (Z - c[0] * x_line) / c[1], 'k', label=f'Z*={Z:.2f}')
    #     elif abs(c[0]) > 1e-9:
    #         ax.axvline(Z / c[0], color='k', label=f'Z*={Z:.2f}')

    # Optimal point
    if solution:
        x_opt, y_opt = solution['x1'], solution['x2']
        ax.plot(x_opt, y_opt, 'ro', markersize=8, label='Điểm tối ưu')
        ax.text(x_opt + 0.1, y_opt + 0.1, f"({x_opt:.2f},{y_opt:.2f})")

    # Finalize
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_title('Miền khả thi của bài toán QHTT')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    fig.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    plot_data = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(fig)
    return plot_data

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)