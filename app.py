from flask import Flask, render_template, request, jsonify 
import io
import base64
from simplex import auto_simplex 
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import numpy as np
import os

app = Flask(__name__)

# Cấu hình Matplotlib cho font (nếu cần hiển thị tiếng Việt trong biểu đồ)
# plt.rcParams['font.family'] = 'DejaVu Sans' 
# plt.rcParams['axes.unicode_minus'] = False

@app.route('/', methods=['GET'])
def index():
    """Hiển thị form nhập liệu ban đầu."""
    default_num_vars = 2
    default_num_constraints = 2
    num_vars = request.args.get('num_vars', default_num_vars, type=int)
    num_constraints = request.args.get('num_constraints', default_num_constraints, type=int)
    error = request.args.get('error', None)
    
    # Lấy lại các giá trị đã nhập từ query parameters nếu có lỗi và redirect
    # Điều này giúp người dùng không phải nhập lại toàn bộ
    # Đây là một cách đơn giản, cách phức tạp hơn có thể dùng session
    form_data = {}
    if error: # Chỉ lấy lại dữ liệu nếu có lỗi được truyền về
        for key in request.args:
            if key not in ['num_vars', 'num_constraints', 'error']:
                form_data[key] = request.args.get(key)

    return render_template('index.html', 
                           num_vars=num_vars, 
                           num_constraints=num_constraints, 
                           error=error,
                           form_data=form_data) # Truyền form_data vào template

@app.route('/solve', methods=['POST'])
def solve():
    """Xử lý dữ liệu form, giải bài toán và hiển thị kết quả."""
    num_vars_input = request.form.get('num_vars')
    num_constraints_input = request.form.get('num_constraints')
    
    error_context = {
        'num_vars': num_vars_input if num_vars_input and num_vars_input.isdigit() else 2,
        'num_constraints': num_constraints_input if num_constraints_input and num_constraints_input.isdigit() else 2,
        'form_data': request.form # Giữ lại toàn bộ form data khi có lỗi
    }

    try:
        if not num_vars_input or not num_vars_input.isdigit() or int(num_vars_input) < 1:
            error_context['error'] = "Số lượng biến (n) phải là số nguyên dương (≥1)."
            return render_template('index.html', **error_context)
        if not num_constraints_input or not num_constraints_input.isdigit() or int(num_constraints_input) < 1:
            error_context['error'] = "Số lượng ràng buộc (m) phải là số nguyên dương (≥1)."
            return render_template('index.html', **error_context)

        num_vars = int(num_vars_input)
        num_constraints = int(num_constraints_input)
        
        error_context['num_vars'] = num_vars
        error_context['num_constraints'] = num_constraints

        c = []
        for i in range(num_vars):
            val = request.form.get(f'c{i}')
            if val is None or val.strip() == '':
                error_context['error'] = f"Hệ số mục tiêu c{i+1} bị thiếu hoặc rỗng."
                return render_template('index.html', **error_context)
            try:
                c.append(float(val))
            except ValueError:
                error_context['error'] = f"Hệ số mục tiêu c{i+1} ('{val}') không phải số hợp lệ."
                return render_template('index.html', **error_context)

        A, b_constraints, constraint_types = [], [], [] # Đổi tên b thành b_constraints để tránh nhầm lẫn
        for i in range(num_constraints):
            row = []
            for j in range(num_vars):
                val = request.form.get(f'A{i}_{j}')
                if val is None or val.strip() == '':
                    error_context['error'] = f"Hệ số ràng buộc A[{i+1}][{j+1}] bị thiếu hoặc rỗng."
                    return render_template('index.html', **error_context)
                try:
                    row.append(float(val))
                except ValueError:
                    error_context['error'] = f"Hệ số ràng buộc A[{i+1}][{j+1}] ('{val}') không phải số hợp lệ."
                    return render_template('index.html', **error_context)
            A.append(row)

            val = request.form.get(f'b{i}') # Đây là b của ràng buộc
            if val is None or val.strip() == '':
                error_context['error'] = f"Hằng số vế phải b{i+1} bị thiếu hoặc rỗng."
                return render_template('index.html', **error_context)
            try:
                b_constraints.append(float(val))
            except ValueError:
                error_context['error'] = f"Hằng số vế phải b{i+1} ('{val}') không phải số hợp lệ."
                return render_template('index.html', **error_context)

            constraint_type = request.form.get(f'constraint_type{i}')
            if constraint_type not in ['<=', '>=', '=']:
                error_context['error'] = f"Loại ràng buộc cho dòng {i+1} không hợp lệ."
                return render_template('index.html', **error_context)
            constraint_types.append(constraint_type)

        variable_types = []
        for i in range(num_vars):
            var_type = request.form.get(f'var_type{i}')
            if var_type not in ['>=0', '<=0', 'URS']:
                error_context['error'] = f"Loại biến cho x{i+1} không hợp lệ."
                return render_template('index.html', **error_context)
            variable_types.append(var_type)

        objective = request.form.get('objective_type', 'Max').lower()
        if objective not in ['max', 'min']:
            error_context['error'] = "Loại bài toán (Max/Min) không hợp lệ."
            return render_template('index.html', **error_context)

        print(f"DEBUG: Calling auto_simplex with:")
        print(f"  A: {A}")
        print(f"  b: {b_constraints}")
        print(f"  c: {c}")
        print(f"  constraint_types: {constraint_types}")
        print(f"  objective: {objective}")
        print(f"  variable_types: {variable_types}")

        result = auto_simplex(A, b_constraints, c, constraint_types, objective, variable_types)
        print(f"DEBUG: Result from auto_simplex: {result}")
        
        status_from_result = result.get('status', '').lower() # Lấy status và chuyển sang chữ thường

        if status_from_result.startswith('lỗi'):
            error_context['error'] = result['status']
            return render_template('index.html', **error_context)

        plot_data = None
        # Điều kiện để vẽ biểu đồ: 2 biến và trạng thái bắt đầu bằng "tối ưu" (không phân biệt hoa thường)
        # và có solution
        should_plot = (num_vars == 2 and 
                       result.get('solution') and 
                       status_from_result.startswith('tối ưu'))

        print(f"DEBUG: num_vars = {num_vars}")
        print(f"DEBUG: result.get('solution') is not None: {result.get('solution') is not None}")
        print(f"DEBUG: status_from_result: '{status_from_result}'")
        print(f"DEBUG: Should plot? {should_plot}")


        if should_plot:
            try:
                plot_data = create_plot(A, b_constraints, constraint_types, c, result['solution'], variable_types, objective)
                print("DEBUG: Plot created successfully.")
            except Exception as plot_error:
                print(f"LỖI KHI TẠO BIỂU ĐỒ: {plot_error}")
                app.logger.error(f"Lỗi khi tạo biểu đồ: {plot_error}", exc_info=True)
                result['plot_error_message'] = f"Không thể tạo biểu đồ: {str(plot_error)[:100]}..." # Giới hạn độ dài thông báo lỗi

        return render_template('result.html', result=result, plot_data=plot_data)

    except Exception as e:
        app.logger.error(f"Lỗi không xác định trong /solve: {e}", exc_info=True)
        error_context['error'] = f"Đã xảy ra lỗi không mong muốn. Vui lòng kiểm tra console của server. ({str(e)[:50]}...)"
        return render_template('index.html', **error_context)


def create_plot(A_orig, b_orig, constraint_types_orig, c_orig, solution, variable_types, objective_type):
    """Tạo biểu đồ miền khả thi và điểm tối ưu cho bài toán 2 biến."""
    
    A = [list(row) for row in A_orig]
    b = list(b_orig) # Đổi tên biến này cho nhất quán
    constraint_types = list(constraint_types_orig)
    c = list(c_orig)

    plt.rc('font', family='DejaVu Sans') 
    plt.rcParams['axes.unicode_minus'] = False

    fig, ax = plt.subplots(figsize=(10, 8))

    plot_points_x = [0.0] # Khởi tạo với float
    plot_points_y = [0.0]
    if solution:
        plot_points_x.append(float(solution.get('x1', 0.0)))
        plot_points_y.append(float(solution.get('x2', 0.0)))

    for i in range(len(A)):
        if abs(A[i][1]) > 1e-9: 
            plot_points_y.append(b[i] / A[i][1])
            plot_points_x.append(0.0)
        if abs(A[i][0]) > 1e-9: 
            plot_points_x.append(b[i] / A[i][0])
            plot_points_y.append(0.0)
    
    for i in range(len(A)):
        for j in range(i + 1, len(A)):
            matrix_A_intersect = np.array([A[i], A[j]])
            vector_b_intersect = np.array([b[i], b[j]])
            try:
                if abs(np.linalg.det(matrix_A_intersect)) > 1e-9: # Kiểm tra định thức khác 0
                    intersect_pt = np.linalg.solve(matrix_A_intersect, vector_b_intersect)
                    plot_points_x.append(intersect_pt[0])
                    plot_points_y.append(intersect_pt[1])
            except np.linalg.LinAlgError:
                continue 

    if not plot_points_x or not plot_points_y: # Xử lý trường hợp list rỗng
        print("CẢNH BÁO: Không có đủ điểm để xác định giới hạn biểu đồ.")
        x_plot_min, x_plot_max = -5, 5 # Giá trị mặc định
        y_plot_min, y_plot_max = -5, 5
    else:
        x_min_data, x_max_data = min(plot_points_x), max(plot_points_x)
        y_min_data, y_max_data = min(plot_points_y), max(plot_points_y)
        
        margin_x = max(1.0, abs(x_max_data - x_min_data) * 0.25) # Tăng margin
        margin_y = max(1.0, abs(y_max_data - y_min_data) * 0.25)

        x_plot_min = x_min_data - margin_x
        x_plot_max = x_max_data + margin_x
        y_plot_min = y_min_data - margin_y
        y_plot_max = y_max_data + margin_y
    
    if '>=0' in variable_types[0] and x_plot_min < -margin_x/3 : x_plot_min = -margin_x/3
    if '>=0' in variable_types[1] and y_plot_min < -margin_y/3 : y_plot_min = -margin_y/3

    x_line_vals = np.linspace(x_plot_min, x_plot_max, 400)
    for i in range(len(A)):
        a1, a2 = A[i][0], A[i][1]
        bi = b[i]
        
        label_terms = []
        if abs(a1) > 1e-9: label_terms.append(f"{a1:g}x₁")
        if abs(a2) > 1e-9:
            if a2 > 0 and label_terms: label_terms.append(f"+ {a2:g}x₂")
            elif a2 < 0 and label_terms: label_terms.append(f"- {abs(a2):g}x₂")
            else: label_terms.append(f"{a2:g}x₂")

        if not label_terms: label_terms.append("0")
        
        op_map = {'<=': '≤', '>=': '≥', '=': '='}
        label = f"{' '.join(label_terms)} {op_map[constraint_types[i]]} {bi:g}"

        if abs(a2) > 1e-9: 
            y_line_vals = (bi - a1 * x_line_vals) / a2
            ax.plot(x_line_vals, y_line_vals, label=label, lw=1.5)
        elif abs(a1) > 1e-9: 
            ax.axvline(x = bi / a1, label=label, lw=1.5)

    if variable_types[0] == '>=0': ax.axvline(x=0, color='gray', linestyle='--', lw=1, label='x₁ ≥ 0')
    if variable_types[0] == '<=0': ax.axvline(x=0, color='gray', linestyle='--', lw=1, label='x₁ ≤ 0')
    if variable_types[1] == '>=0': ax.axhline(y=0, color='gray', linestyle='--', lw=1, label='x₂ ≥ 0')
    if variable_types[1] == '<=0': ax.axhline(y=0, color='gray', linestyle='--', lw=1, label='x₂ ≤ 0')

    X_grid, Y_grid = np.meshgrid(np.linspace(x_plot_min, x_plot_max, 200), 
                                 np.linspace(y_plot_min, y_plot_max, 200))
    feasible_mask = np.ones(X_grid.shape, dtype=bool)

    for i in range(len(A)):
        val = A[i][0] * X_grid + A[i][1] * Y_grid
        if constraint_types[i] == '<=':
            feasible_mask &= (val <= b[i] + 1e-6) 
        elif constraint_types[i] == '>=':
            feasible_mask &= (val >= b[i] - 1e-6)
        elif constraint_types[i] == '=':
            feasible_mask &= (np.abs(val - b[i]) < 1e-3) 

    if variable_types[0] == '>=0': feasible_mask &= (X_grid >= -1e-6)
    if variable_types[0] == '<=0': feasible_mask &= (X_grid <= 1e-6)
    if variable_types[1] == '>=0': feasible_mask &= (Y_grid >= -1e-6)
    if variable_types[1] == '<=0': feasible_mask &= (Y_grid <= 1e-6)
    
    ax.imshow(feasible_mask.astype(int), extent=(x_plot_min, x_plot_max, y_plot_min, y_plot_max),
              origin='lower', cmap="Greens", alpha=0.3, aspect='auto')

    if solution and 'x1' in solution and 'x2' in solution:
        x_opt, y_opt = float(solution['x1']), float(solution['x2']) # Đảm bảo là float
        ax.plot(x_opt, y_opt, 'o', color='red', markersize=10, label=f'Tối ưu: ({x_opt:.2f}, {y_opt:.2f})', zorder=5)
        if objective_type in ['max', 'min'] and (abs(c[0]) > 1e-9 or abs(c[1]) > 1e-9) :
            z_opt = c[0]*x_opt + c[1]*y_opt
            if abs(c[1]) > 1e-9: 
                y_obj_line = (z_opt - c[0]*x_line_vals) / c[1]
            elif abs(c[0]) > 1e-9: 
                 ax.axvline(x = z_opt / c[0], linestyle='--', color='purple', lw=1.5, label=f'Đường mục tiêu z={z_opt:.2f}')

    ax.set_xlabel("x₁", fontsize=12)
    ax.set_ylabel("x₂", fontsize=12)
    ax.set_title("Biểu đồ miền khả thi và điểm tối ưu", fontsize=14)
    ax.grid(True, linestyle=':', alpha=0.6)
    
    # Điều chỉnh vị trí legend để không che biểu đồ
    handles, labels = ax.get_legend_handles_labels()
    if handles: # Chỉ hiển thị legend nếu có gì để hiển thị
        ax.legend(handles, labels, fontsize=9, loc='center left', bbox_to_anchor=(1.01, 0.5))

    ax.set_xlim(x_plot_min, x_plot_max)
    ax.set_ylim(y_plot_min, y_plot_max)

    fig.tight_layout(rect=[0, 0, 0.80, 1]) # Điều chỉnh rect để có không gian cho legend

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100) 
    buf.seek(0)
    plot_data_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(fig) 
    
    return plot_data_base64

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000)) 
    app.run(host='0.0.0.0', port=port, debug=os.environ.get('FLASK_DEBUG', 'False').lower() == 'true')
