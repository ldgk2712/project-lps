from flask import Flask, render_template, request, jsonify
import io
import base64
from simplex_bland import auto_simplex # Đảm bảo import từ file simplex đã cập nhật
import matplotlib
matplotlib.use('Agg') # Chạy matplotlib ở chế độ không GUI
import matplotlib.pyplot as plt
import numpy as np
import os
from fractions import Fraction

app = Flask(__name__)

# Sử dụng hằng số tolerance từ simplex.py nếu cần, hoặc định nghĩa một hằng số tương tự ở đây
APP_TOLERANCE = 1e-9 

def parse_number(val: str) -> float:
    """Chuyển đổi chuỗi đầu vào thành số float, hỗ trợ cả dạng phân số."""
    if not val or val.strip() == '':
        raise ValueError("Đầu vào rỗng")
    try:
        return float(val)
    except ValueError:
        try:
            # Thử chuyển đổi từ dạng a/b
            return float(Fraction(val))
        except (ValueError, ZeroDivisionError):
            raise ValueError(f"Định dạng số không hợp lệ: '{val}'")

@app.route('/', methods=['GET'])
def index():
    """Hiển thị trang nhập liệu chính."""
    default_num_vars = 2
    default_num_constraints = 2
    num_vars = request.args.get('num_vars', default_num_vars, type=int)
    num_constraints = request.args.get('num_constraints', default_num_constraints, type=int)
    error = request.args.get('error', None)
    
    # Khôi phục dữ liệu form nếu có lỗi để người dùng không phải nhập lại
    form_data = {}
    if error:
        for key in request.args:
            if key not in ['num_vars', 'num_constraints', 'error']:
                form_data[key] = request.args.get(key)
                
    return render_template('index.html', 
                           num_vars=num_vars, 
                           num_constraints=num_constraints,
                           error=error,
                           form_data=form_data)

@app.route('/solve', methods=['POST'])
def solve():
    """Xử lý yêu cầu giải bài toán QHTT."""
    num_vars_input = request.form.get('num_vars')
    num_constraints_input = request.form.get('num_constraints')

    # Chuẩn bị context để render lại form nếu có lỗi
    error_context = {
        'num_vars': num_vars_input if num_vars_input and num_vars_input.isdigit() else 2,
        'num_constraints': num_constraints_input if num_constraints_input and num_constraints_input.isdigit() else 2,
        'form_data': request.form # Giữ lại toàn bộ dữ liệu form
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

        c = [] # Hệ số hàm mục tiêu
        for i in range(num_vars):
            val = request.form.get(f'c{i}')
            if val is None or val.strip() == '':
                error_context['error'] = f"Hệ số mục tiêu c{i+1} bị thiếu hoặc rỗng."
                return render_template('index.html', **error_context)
            try:
                c.append(parse_number(val))
            except ValueError:
                error_context['error'] = f"Hệ số mục tiêu c{i+1} ('{val}') không phải số hợp lệ."
                return render_template('index.html', **error_context)

        A, b_constraints, constraint_types = [], [], [] # Ma trận A, vector b, loại ràng buộc
        for i in range(num_constraints):
            row = []
            for j in range(num_vars):
                val = request.form.get(f'A{i}_{j}')
                if val is None or val.strip() == '':
                    error_context['error'] = f"Hệ số ràng buộc A[{i+1}][{j+1}] bị thiếu hoặc rỗng."
                    return render_template('index.html', **error_context)
                try:
                    row.append(parse_number(val))
                except ValueError:
                    error_context['error'] = f"Hệ số ràng buộc A[{i+1}][{j+1}] ('{val}') không phải số hợp lệ."
                    return render_template('index.html', **error_context)
            A.append(row)

            val = request.form.get(f'b{i}')
            if val is None or val.strip() == '':
                error_context['error'] = f"Hằng số vế phải b{i+1} bị thiếu hoặc rỗng."
                return render_template('index.html', **error_context)
            try:
                b_constraints.append(parse_number(val))
            except ValueError:
                error_context['error'] = f"Hằng số vế phải b{i+1} ('{val}') không phải số hợp lệ."
                return render_template('index.html', **error_context)
            
            constraint_type = request.form.get(f'constraint_type{i}')
            if constraint_type not in ['<=', '>=', '=']:
                error_context['error'] = f"Loại ràng buộc cho dòng {i+1} không hợp lệ."
                return render_template('index.html', **error_context)
            constraint_types.append(constraint_type)

        variable_types = [] # Loại biến (>=0, <=0, URS)
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

        # Gọi hàm giải Simplex
        result = auto_simplex(A, b_constraints, c, constraint_types, objective, variable_types)
        
        # Lấy trạng thái từ kết quả để kiểm tra
        status_from_result_str = result.get('status', 'Lỗi (Error)') # Mặc định là lỗi nếu không có status

        if status_from_result_str.lower().startswith('lỗi'):
            error_context['error'] = result.get('error_message', status_from_result_str)
            # Thêm các bước giải (nếu có) vào context để debug
            error_context['simplex_steps'] = result.get('steps') 
            return render_template('index.html', **error_context)

        plot_data = None
        # Điều kiện để vẽ đồ thị: chỉ vẽ khi có 2 biến
        # và trạng thái không phải là lỗi.
        # Việc có vẽ điểm tối ưu hay không sẽ do create_plot quyết định dựa trên solution và status.
        should_try_plotting = (num_vars == 2 and not status_from_result_str.lower().startswith('lỗi'))
        
        if should_try_plotting:
            try:
                # Truyền cả solution và status vào hàm tạo đồ thị
                plot_data = create_plot(
                    A, b_constraints, constraint_types, 
                    result.get('solution', {}), # Truyền solution, có thể rỗng
                    variable_types, 
                    status_from_result_str # Truyền chuỗi trạng thái
                )
            except Exception as plot_error:
                app.logger.error(f"Lỗi khi tạo biểu đồ: {plot_error}", exc_info=True)
                result['plot_error_message'] = f"Không thể tạo biểu đồ: {str(plot_error)[:100]}..."
        
        return render_template('result.html', result=result, plot_data=plot_data)

    except ValueError as ve: # Lỗi từ parse_number hoặc các kiểm tra đầu vào khác
        app.logger.warning(f"Lỗi giá trị đầu vào: {ve}")
        error_context['error'] = str(ve)
        return render_template('index.html', **error_context)
    except Exception as e:
        app.logger.error(f"Lỗi không xác định trong /solve: {e}", exc_info=True)
        error_context['error'] = f"Đã xảy ra lỗi không mong muốn. Vui lòng kiểm tra console của server. ({str(e)[:100]}...)"
        return render_template('index.html', **error_context)


def create_plot(A_orig, b_orig, constraint_types_orig, solution, variable_types, status_str):
    """
    Tạo biểu đồ miền khả thi và điểm tối ưu (nếu có) cho bài toán 2 biến.
    solution: Dict chứa nghiệm {'x1': val1, 'x2': val2} hoặc rỗng.
    status_str: Chuỗi trạng thái từ hàm giải Simplex (ví dụ: 'Tối ưu (Optimal)').
    """
    plt.rc('font', family='DejaVu Sans') # Đảm bảo font hỗ trợ Unicode
    plt.rcParams['axes.unicode_minus'] = False # Hiển thị dấu trừ đúng cách
    fig, ax = plt.subplots(figsize=(10, 8))

    # Xác định xem có nên vẽ điểm tối ưu không
    # Dựa vào việc 'solution' có giá trị và trạng thái là 'Tối ưu' hoặc 'Vô số nghiệm'
    has_valid_solution_for_plot = False
    x_opt, y_opt = None, None
    if solution and (status_str.lower().startswith('tối ưu') or status_str.lower().startswith('vô số nghiệm')):
        try:
            x1_val_str = solution.get('x1')
            x2_val_str = solution.get('x2')
            if x1_val_str is not None and x2_val_str is not None:
                x_opt = float(x1_val_str) # Giá trị đã được format_expression_for_printing
                y_opt = float(x2_val_str)
                has_valid_solution_for_plot = True
        except (ValueError, TypeError):
            # Nếu không chuyển đổi được (ví dụ nghiệm là tham số cho 'Vô số nghiệm'), không vẽ điểm
            has_valid_solution_for_plot = False
            app.logger.warning(f"Không thể chuyển đổi nghiệm tối ưu sang số để vẽ: x1='{solution.get('x1')}', x2='{solution.get('x2')}'")


    # Tạo lưới điểm để vẽ miền khả thi
    # Xác định giới hạn cho các trục dựa trên giao điểm của các ràng buộc và điểm tối ưu (nếu có)
    plot_points_x, plot_points_y = [0.0], [0.0] # Bắt đầu với gốc tọa độ
    if has_valid_solution_for_plot:
        plot_points_x.append(x_opt)
        plot_points_y.append(y_opt)

    for i in range(len(A_orig)):
        # Giao điểm với trục x2 (x1=0)
        if abs(A_orig[i][1]) > APP_TOLERANCE:
            plot_points_y.append(b_orig[i] / A_orig[i][1])
            plot_points_x.append(0.0)
        # Giao điểm với trục x1 (x2=0)
        if abs(A_orig[i][0]) > APP_TOLERANCE:
            plot_points_x.append(b_orig[i] / A_orig[i][0])
            plot_points_y.append(0.0)

    for i in range(len(A_orig)):
        for j in range(i + 1, len(A_orig)):
            # Giải hệ A_intersect * X = b_intersect
            matrix_A_intersect = np.array([A_orig[i], A_orig[j]])
            vector_b_intersect = np.array([b_orig[i], b_orig[j]])
            if abs(np.linalg.det(matrix_A_intersect)) > APP_TOLERANCE: # Kiểm tra tính khả nghịch
                try:
                    intersect_pt = np.linalg.solve(matrix_A_intersect, vector_b_intersect)
                    plot_points_x.append(intersect_pt[0])
                    plot_points_y.append(intersect_pt[1])
                except np.linalg.LinAlgError:
                    pass # Bỏ qua nếu không giải được (song song/trùng)
    
    # Giới hạn dữ liệu thô
    x_min_data = min(plot_points_x) if plot_points_x else -1
    x_max_data = max(plot_points_x) if plot_points_x else 1
    y_min_data = min(plot_points_y) if plot_points_y else -1
    y_max_data = max(plot_points_y) if plot_points_y else 1

    # Lề cho tính toán (khá rộng để không bỏ sót miền khả thi)
    margin_x = max(2.0, abs(x_max_data - x_min_data) * 0.5) 
    margin_y = max(2.0, abs(y_max_data - y_min_data) * 0.5)

    # Giới hạn tính toán ban đầu
    calc_x_min, calc_x_max = x_min_data - margin_x, x_max_data + margin_x
    calc_y_min, calc_y_max = y_min_data - margin_y, y_max_data + margin_y
    
    # Điều chỉnh giới hạn dựa trên loại biến (>=0 hoặc <=0)
    # Cho phép một chút không gian âm/dương nhỏ để trực quan hóa trục
    small_axis_offset = 0.1 
    if variable_types[0] == '>=0' and calc_x_min < -small_axis_offset * margin_x : calc_x_min = -small_axis_offset * margin_x
    elif variable_types[0] == '<=0' and calc_x_max > small_axis_offset * margin_x : calc_x_max = small_axis_offset * margin_x
    
    if variable_types[1] == '>=0' and calc_y_min < -small_axis_offset * margin_y : calc_y_min = -small_axis_offset * margin_y
    elif variable_types[1] == '<=0' and calc_y_max > small_axis_offset * margin_y : calc_y_max = small_axis_offset * margin_y

    # Đảm bảo khoảng cách tối thiểu nếu các điểm rất gần nhau
    if abs(calc_x_max - calc_x_min) < 1.0: calc_x_min -=0.5; calc_x_max +=0.5
    if abs(calc_y_max - calc_y_min) < 1.0: calc_y_min -=0.5; calc_y_max +=0.5
    
    # Tạo lưới điểm cho imshow
    grid_points = 250 # Độ chi tiết của lưới
    x_grid = np.linspace(calc_x_min, calc_x_max, grid_points)
    y_grid = np.linspace(calc_y_min, calc_y_max, grid_points)
    X_mesh, Y_mesh = np.meshgrid(x_grid, y_grid)

    # Xác định miền khả thi trên lưới
    feasible_mask = np.ones(X_mesh.shape, dtype=bool)
    for i in range(len(A_orig)):
        val = A_orig[i][0] * X_mesh + A_orig[i][1] * Y_mesh
        # Sử dụng APP_TOLERANCE cho so sánh
        if constraint_types_orig[i] == '<=': feasible_mask &= (val <= b_orig[i] + APP_TOLERANCE)
        elif constraint_types_orig[i] == '>=': feasible_mask &= (val >= b_orig[i] - APP_TOLERANCE)
        elif constraint_types_orig[i] == '=': feasible_mask &= (np.abs(val - b_orig[i]) < APP_TOLERANCE)

    # Áp dụng ràng buộc loại biến
    if variable_types[0] == '>=0': feasible_mask &= (X_mesh >= -APP_TOLERANCE)
    elif variable_types[0] == '<=0': feasible_mask &= (X_mesh <= APP_TOLERANCE)
    if variable_types[1] == '>=0': feasible_mask &= (Y_mesh >= -APP_TOLERANCE)
    elif variable_types[1] == '<=0': feasible_mask &= (Y_mesh <= APP_TOLERANCE)

    # Vẽ miền khả thi
    ax.imshow(feasible_mask.astype(int), extent=(calc_x_min, calc_x_max, calc_y_min, calc_y_max),
              origin='lower', cmap="Greens", alpha=0.3, aspect='auto')

    # Vẽ các đường thẳng ràng buộc
    line_plot_x_vals = np.array([calc_x_min, calc_x_max]) # Dùng array để tính toán dễ hơn
    for i in range(len(A_orig)):
        a1, a2 = A_orig[i][0], A_orig[i][1]
        bi = b_orig[i]
        
        label_terms = []
        if abs(a1) > APP_TOLERANCE: label_terms.append(f"{a1:g}x₁")
        if abs(a2) > APP_TOLERANCE:
            op_str = " + " if a2 > 0 and label_terms else (" - " if a2 < 0 and label_terms else "")
            val_str = f"{abs(a2):g}" if abs(abs(a2)-1.0) > APP_TOLERANCE or not label_terms else ""
            if not label_terms and a2 < 0 : op_str = "-" # Trường hợp -x2
            label_terms.append(f"{op_str}{val_str}x₂" if label_terms else f"{a2:g}x₂")

        if not label_terms: label_terms.append("0") 
        op_map = {'<=': '≤', '>=': '≥', '=': '='}
        label = f"{''.join(label_terms)} {op_map[constraint_types_orig[i]]} {bi:g}"

        if abs(a2) > APP_TOLERANCE: # Đường không thẳng đứng
            y_line_vals = (bi - a1 * line_plot_x_vals) / a2
            ax.plot(line_plot_x_vals, y_line_vals, label=label, lw=1.5)
        elif abs(a1) > APP_TOLERANCE: # Đường thẳng đứng
            ax.axvline(x=bi / a1, label=label, lw=1.5)

    # Vẽ đường thẳng dấu của biến (nếu cần)
    if variable_types[0] == '>=0': ax.axvline(x=0, color='gray', linestyle='--', lw=1, label='x₁ ≥ 0')
    elif variable_types[0] == '<=0': ax.axvline(x=0, color='gray', linestyle='--', lw=1, label='x₁ ≤ 0')
    if variable_types[1] == '>=0': ax.axhline(y=0, color='gray', linestyle='--', lw=1, label='x₂ ≥ 0')
    elif variable_types[1] == '<=0': ax.axhline(y=0, color='gray', linestyle='--', lw=1, label='x₂ ≤ 0')

    # Vẽ điểm tối ưu nếu có và hợp lệ
    if has_valid_solution_for_plot:
        ax.plot(x_opt, y_opt, 'o', color='red', markersize=8, markeredgecolor='black',
                label=f'Tối ưu: ({x_opt:.2f}, {y_opt:.2f})', zorder=5)

    # Đặt tiêu đề và nhãn
    plot_title = "Biểu đồ miền khả thi"
    if has_valid_solution_for_plot:
        plot_title += " và điểm tối ưu"
    # Thêm thông tin trạng thái vào tiêu đề nếu không phải tối ưu/vô số nghiệm
    elif status_str.lower().startswith('vô nghiệm'):
        plot_title += " (Vô nghiệm)"
    elif status_str.lower().startswith('không giới nội'):
        plot_title += " (Không giới nội)"
    # Các trạng thái khác có thể được thêm vào đây nếu cần

    ax.set_title(plot_title, fontsize=14)
    ax.set_xlabel("x₁", fontsize=12)
    ax.set_ylabel("x₂", fontsize=12)
    ax.grid(True, linestyle=':', alpha=0.7)
    
    # Xử lý legend
    handles, labels_legend = ax.get_legend_handles_labels()
    if handles: # Chỉ hiển thị legend nếu có gì để hiển thị
        ax.legend(handles, labels_legend, fontsize=9, loc='center left', bbox_to_anchor=(1.01, 0.5))
    
    # Đặt giới hạn cuối cùng cho plot
    ax.set_xlim(calc_x_min, calc_x_max)
    ax.set_ylim(calc_y_min, calc_y_max)

    fig.tight_layout(rect=[0, 0, 0.82, 1]) # Điều chỉnh để legend không bị cắt (0.80 -> 0.82)

    # Lưu biểu đồ vào buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100) # dpi=100 là đủ cho web
    buf.seek(0)
    plot_data_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(fig) # Đóng figure để giải phóng bộ nhớ

    return plot_data_base64


if __name__ == '__main__':
    # Lấy port từ biến môi trường hoặc dùng mặc định 5000
    port = int(os.environ.get('PORT', 5000))
    # Chạy app ở chế độ debug nếu FLASK_DEBUG=True
    app.run(host='0.0.0.0', port=port, debug=os.environ.get('FLASK_DEBUG', 'True').lower() == 'true')
