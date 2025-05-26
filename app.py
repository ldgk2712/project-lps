from flask import Flask, render_template, request, send_file
import io
import base64
from simplex import solve_simplex
import matplotlib.pyplot as plt

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Lấy dữ liệu từ form
        num_vars = int(request.form['num_vars'])
        num_constraints = int(request.form['num_constraints'])
        objective_type = request.form['objective_type']
        
        # Hệ số hàm mục tiêu
        c = [float(request.form[f'c{i}']) for i in range(num_vars)]
        
        # Ma trận A và vector b
        A = []
        b = []
        constraint_types = []
        for i in range(num_constraints):
            row = [float(request.form[f'A{i}_{j}']) for j in range(num_vars)]
            A.append(row)
            b.append(float(request.form[f'b{i}']))
            constraint_types.append(request.form[f'constraint_type{i}'])
        
        # Loại biến
        variable_types = [request.form[f'var_type{i}'] for i in range(num_vars)]
        
        # Giải bài toán
        result = solve_simplex(A, b, c, constraint_types, objective_type, variable_types)
        
        # Tạo hình ảnh hóa nếu num_vars == 2
        plot_data = None
        if num_vars == 2 and result['status'] == 'Tối ưu (Optimal)':
            plot_data = create_plot(A, b, constraint_types, c, objective_type, result['solution'])
        
        return render_template('result.html', result=result, plot_data=plot_data)
    
    return render_template('index.html', num_vars=2, num_constraints=2)

def create_plot(A, b, constraint_types, c, objective_type, solution):
    plt.figure(figsize=(6, 4))
    
    # Vẽ miền khả thi
    x = range(0, 10)
    for i in range(len(A)):
        a1, a2 = A[i][0], A[i][1]
        b_val = b[i]
        if abs(a2) > 1e-9:
            y = [(b_val - a1 * xi) / a2 for xi in x]
            plt.plot(x, y, label=f'Ràng buộc {i+1}')
    
    # Vẽ điểm tối ưu
    if solution:
        plt.plot(solution['x1'], solution['x2'], 'ro', label='Điểm tối ưu')
        plt.text(solution['x1'], solution['x2'], f"({solution['x1']:.2f}, {solution['x2']:.2f})")
    
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.grid(True)
    
    # Lưu biểu đồ vào bộ nhớ
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plot_data = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()
    
    return plot_data

import os
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)