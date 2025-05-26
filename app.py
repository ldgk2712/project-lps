from flask import Flask, render_template, request, send_file
import io
import base64
from simplex import solve_simplex
import matplotlib.pyplot as plt

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    default_num_vars = 2
    default_num_constraints = 2
    
    if request.method == 'POST':
        try:
            num_vars_input = request.form.get('num_vars', default_num_vars)
            num_constraints_input = request.form.get('num_constraints', default_num_constraints)
            
            if not num_vars_input.isdigit() or int(num_vars_input) < 1:
                return render_template('index.html', 
                                     num_vars=num_vars_input, 
                                     num_constraints=num_constraints_input, 
                                     error="Số lượng biến (n) phải là số nguyên dương (≥ 1).")
            if not num_constraints_input.isdigit() or int(num_constraints_input) < 1:
                return render_template('index.html', 
                                     num_vars=num_vars_input, 
                                     num_constraints=num_constraints_input, 
                                     error="Số lượng ràng buộc (m) phải là số nguyên dương (≥ 1).")
            
            num_vars = int(num_vars_input)
            num_constraints = int(num_constraints_input)
            
            if 'update' in request.form:
                return render_template('index.html', num_vars=num_vars, num_constraints=num_constraints)
            
            c = [float(request.form[f'c{i}']) for i in range(num_vars)]
            A = []
            b = []
            constraint_types = []
            for i in range(num_constraints):
                row = [float(request.form[f'A{i}_{j}']) for j in range(num_vars)]
                A.append(row)
                b.append(float(request.form[f'b{i}']))
                constraint_types.append(request.form[f'constraint_type{i}'])
            variable_types = [request.form[f'var_type{i}'] for i in range(num_vars)]
            
            result = solve_simplex(A, b, c, constraint_types, request.form['objective_type'], variable_types)
            
            if result['status'].startswith('Lỗi'):
                return render_template('index.html', 
                                     num_vars=num_vars, 
                                     num_constraints=num_constraints, 
                                     error=result['status'])
            
            plot_data = None
            if num_vars == 2 and result['status'] == 'Tối ưu (Optimal)':
                plot_data = create_plot(A, b, constraint_types, c, request.form['objective_type'], result['solution'])
            
            return render_template('result.html', result=result, plot_data=plot_data)
        
        except (ValueError, KeyError) as e:
            return render_template('index.html', 
                                 num_vars=num_vars_input, 
                                 num_constraints=num_constraints_input, 
                                 error="Vui lòng nhập đầy đủ và đúng định dạng số.")
    
    return render_template('index.html', num_vars=default_num_vars, num_constraints=default_num_constraints)

def create_plot(A, b, constraint_types, c, objective_type, solution):
    plt.figure(figsize=(6, 4))
    x = range(0, 20)  # Tăng phạm vi để phù hợp với bài toán
    for i in range(len(A)):
        a1, a2 = A[i][0], A[i][1]
        b_val = b[i]
        if abs(a2) > 1e-9:
            y = [(b_val - a1 * xi) / a2 for xi in x]
            plt.plot(x, y, label=f'Ràng buộc {i+1}')
        elif abs(a1) > 1e-9:
            x_val = b_val / a1
            plt.axvline(x=x_val, label=f'Ràng buộc {i+1}')
    if solution:
        plt.plot(solution['x1'], solution['x2'], 'ro', label='Điểm tối ưu')
        plt.text(solution['x1'], solution['x2'], f"({solution['x1']:.2f}, {solution['x2']:.2f})")
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.grid(True)
    plt.xlim(0, 20)
    plt.ylim(0, 20)
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plot_data = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()
    return plot_data

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)