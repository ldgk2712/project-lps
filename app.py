# app.py
from flask import Flask, render_template, request, jsonify
import io
import base64
from simplex_bland import auto_simplex # Simplex với Quy tắc Bland
from simplex_two_phase import simplex_two_phase # Simplex 2 Pha
import matplotlib
matplotlib.use('Agg') # Chạy matplotlib ở chế độ không GUI
import matplotlib.pyplot as plt
import numpy as np
import os
from fractions import Fraction

app = Flask(__name__)

APP_TOLERANCE = 1e-9

def parse_number(val: str) -> float:
    """Chuyển đổi chuỗi đầu vào thành số float, hỗ trợ cả dạng phân số."""
    if not val or val.strip() == '':
        raise ValueError("Đầu vào rỗng")
    try:
        return float(val)
    except ValueError:
        try:
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

    error_context = {
        'num_vars': num_vars_input if num_vars_input and num_vars_input.isdigit() else 2,
        'num_constraints': num_constraints_input if num_constraints_input and num_constraints_input.isdigit() else 2,
        'form_data': request.form
    }

    try:
        # --- (Phần parsing và validation giữ nguyên) ---
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

        c_coeffs = []
        for i in range(num_vars):
            val = request.form.get(f'c{i}')
            if val is None or val.strip() == '':
                error_context['error'] = f"Hệ số mục tiêu c{i+1} bị thiếu hoặc rỗng."
                return render_template('index.html', **error_context)
            try:
                c_coeffs.append(parse_number(val))
            except ValueError:
                error_context['error'] = f"Hệ số mục tiêu c{i+1} ('{val}') không phải số hợp lệ."
                return render_template('index.html', **error_context)

        A_matrix, b_vector, constraint_types_list = [], [], []
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
            A_matrix.append(row)

            val = request.form.get(f'b{i}')
            if val is None or val.strip() == '':
                error_context['error'] = f"Hằng số vế phải b{i+1} bị thiếu hoặc rỗng."
                return render_template('index.html', **error_context)
            try:
                b_vector.append(parse_number(val))
            except ValueError:
                error_context['error'] = f"Hằng số vế phải b{i+1} ('{val}') không phải số hợp lệ."
                return render_template('index.html', **error_context)

            constraint_type = request.form.get(f'constraint_type{i}')
            if constraint_type not in ['<=', '>=', '=']:
                error_context['error'] = f"Loại ràng buộc cho dòng {i+1} không hợp lệ."
                return render_template('index.html', **error_context)
            constraint_types_list.append(constraint_type)

        variable_types_list = []
        for i in range(num_vars):
            var_type = request.form.get(f'var_type{i}')
            if var_type not in ['>=0', '<=0', 'URS']:
                error_context['error'] = f"Loại biến cho x{i+1} không hợp lệ."
                return render_template('index.html', **error_context)
            variable_types_list.append(var_type)

        objective_str = request.form.get('objective_type', 'Max').lower()
        if objective_str not in ['max', 'min']:
            error_context['error'] = "Loại bài toán (Max/Min) không hợp lệ."
            return render_template('index.html', **error_context)
        # --- (Kết thúc phần parsing) ---

        app.logger.info("Giải bằng phương pháp Đơn hình (Quy tắc Bland)...")
        try:
            result_bland = auto_simplex(
                A_matrix, b_vector, c_coeffs, constraint_types_list,
                objective_str, variable_types_list
            )
            app.logger.info(f"Kết quả Đơn hình (Bland): {result_bland.get('status')}")
        except Exception as e_bland:
            app.logger.error(f"Lỗi khi chạy auto_simplex (Bland): {e_bland}", exc_info=True)
            result_bland = {
                'status': 'Lỗi (Error)', 'z': "N/A", 'solution': {}, 'steps': {},
                'error_message': f"Lỗi khi thực thi Đơn hình Bland: {str(e_bland)}",
                'parameter_conditions': ""
            }

        app.logger.info("Giải bằng phương pháp Đơn hình 2 Pha...")
        result_two_phase = simplex_two_phase(
            A_orig=A_matrix, b_orig=b_vector, c_orig=c_coeffs,
            constraint_types_orig=constraint_types_list,
            objective_type_orig=objective_str,
            variable_types_orig=variable_types_list
        )
        app.logger.info(f"Kết quả Đơn hình 2 Pha: {result_two_phase.get('status')}")

        original_steps_bland = result_bland.get('steps') if result_bland else None
        original_steps_two_phase = result_two_phase.get('steps') if result_two_phase else None

        # Hiển thị kết quả
        plot_data_base64 = None
        plot_error_message = None
        if num_vars == 2:
            app.logger.info("Tạo đồ thị cho bài toán 2 biến...")
            solution_for_plot = {}
            status_for_plot = "Lỗi (Error)"
            steps_for_plot = None

            # Ưu tiên kết quả từ 2 Pha vì tổng quát hơn
            if result_two_phase and (result_two_phase.get('status', '').lower().startswith('tối ưu') or result_two_phase.get('status', '').lower().startswith('vô số nghiệm')):
                solution_for_plot = result_two_phase.get('solution', {})
                status_for_plot = result_two_phase.get('status')
                # Lấy steps từ 2-Pha để vẽ đường đi nếu có
                steps_for_plot = result_two_phase.get('steps')
            elif result_bland and (result_bland.get('status', '').lower().startswith('tối ưu') or result_bland.get('status', '').lower().startswith('vô số nghiệm')):
                solution_for_plot = result_bland.get('solution', {})
                status_for_plot = result_bland.get('status')
                steps_for_plot = result_bland.get('steps')

            try:
                plot_data_base64 = create_plot(
                    A_matrix, b_vector, constraint_types_list,
                    solution_for_plot,
                    variable_types_list,
                    status_for_plot,
                    steps_history_for_plot=steps_for_plot
                )
                app.logger.info("Tạo đồ thị thành công.")
            except Exception as plot_error:
                app.logger.error(f"Lỗi khi tạo biểu đồ: {plot_error}", exc_info=True)
                plot_error_message = f"Không thể tạo biểu đồ: {str(plot_error)[:150]}..."

        return render_template('result.html',
                               result_bland=result_bland,
                               result_two_phase=result_two_phase,
                               plot_data=plot_data_base64,
                               plot_error_message=plot_error_message,
                               num_vars=num_vars)

    except ValueError as ve:
        app.logger.warning(f"Lỗi giá trị đầu vào: {ve}")
        error_context['error'] = str(ve)
        return render_template('index.html', **error_context)
    except Exception as e:
        app.logger.error(f"Lỗi không xác định trong /solve: {e}", exc_info=True)
        error_context['error'] = f"Đã xảy ra lỗi không mong muốn. Vui lòng kiểm tra console của server. ({str(e)[:100]}...)"
        return render_template('index.html', **error_context)


def create_plot(A_orig, b_orig, constraint_types_orig, solution, variable_types, status_str, steps_history_for_plot=None):
    """
    Tạo biểu đồ miền khả thi, điểm tối ưu và đường đi Simplex (nếu có) cho bài toán 2 biến.
    """
    plt.rc('font', family='DejaVu Sans')
    plt.rcParams['axes.unicode_minus'] = False
    fig, ax = plt.subplots(figsize=(10, 8))

    has_valid_solution_for_plot = False
    x_opt, y_opt = None, None
    
    # === FIX: SỬA LỖI ĐỌC DỮ LIỆU NGHIỆM TỐI ƯU ===
    if solution and solution.get('solution') and (status_str.lower().startswith('tối ưu') or status_str.lower().startswith('vô số nghiệm')):
        try:
            # Chuyển đổi cấu trúc list of dicts thành một dict đơn giản để dễ truy cập
            solution_dict = {item['variable']: item['expression'] for item in solution['solution']}
            
            x1_val_str = solution_dict.get('x1')
            x2_val_str = solution_dict.get('x2')

            if x1_val_str is not None and x2_val_str is not None:
                # Đối với trường hợp vô số nghiệm (biểu thức chứa tham số),
                # ta chỉ vẽ một điểm đại diện bằng cách cố gắng parse hằng số từ biểu thức.
                if isinstance(x1_val_str, str) and any(c.isalpha() for c in x1_val_str):
                     # Lấy phần hằng số (thường là số hạng đầu tiên)
                     x1_val_str = x1_val_str.split()[0]
                if isinstance(x2_val_str, str) and any(c.isalpha() for c in x2_val_str):
                     x2_val_str = x2_val_str.split()[0]

                x_opt = parse_number(str(x1_val_str))
                y_opt = parse_number(str(x2_val_str))
                has_valid_solution_for_plot = True
        except (ValueError, TypeError, IndexError):
            has_valid_solution_for_plot = False
            app.logger.warning(f"Không thể chuyển đổi nghiệm tối ưu sang số để vẽ: solution data='{solution}'")
    # === END FIX ===

    plot_points_x, plot_points_y = [], []
    if steps_history_for_plot:
        plot_points_x.append(0.0)
        plot_points_y.append(0.0)

    if has_valid_solution_for_plot:
        plot_points_x.append(x_opt)
        plot_points_y.append(y_opt)

    # Lấy tọa độ từ các bước lặp để xác định phạm vi của biểu đồ
    if steps_history_for_plot:
        for step_data in steps_history_for_plot.values():
            # Cấu trúc của steps_history từ simplex_bland và simplex_two_phase khác nhau
            coords = None
            if 'coords' in step_data: # Từ simplex_bland
                coords = step_data.get('coords')
            elif 'tableau' in step_data: # Từ simplex_two_phase (cần tính toán)
                # Phần này phức tạp hơn, tạm thời bỏ qua để tập trung vào lỗi chính
                # Nếu cần vẽ đường đi cho 2-Pha, cần thêm logic ở đây
                pass
            
            if coords and len(coords) == 2:
                plot_points_x.append(coords[0])
                plot_points_y.append(coords[1])


    for i in range(len(A_orig)):
        if abs(A_orig[i][1]) > APP_TOLERANCE:
            plot_points_y.append(b_orig[i] / A_orig[i][1])
            plot_points_x.append(0.0)
        if abs(A_orig[i][0]) > APP_TOLERANCE:
            plot_points_x.append(b_orig[i] / A_orig[i][0])
            plot_points_y.append(0.0)

    for i in range(len(A_orig)):
        for j in range(i + 1, len(A_orig)):
            matrix_A_intersect = np.array([A_orig[i], A_orig[j]])
            vector_b_intersect = np.array([b_orig[i], b_orig[j]])
            if abs(np.linalg.det(matrix_A_intersect)) > APP_TOLERANCE:
                try:
                    intersect_pt = np.linalg.solve(matrix_A_intersect, vector_b_intersect)
                    plot_points_x.append(intersect_pt[0])
                    plot_points_y.append(intersect_pt[1])
                except np.linalg.LinAlgError:
                    pass

    x_min_data = min(plot_points_x) if plot_points_x else -1
    x_max_data = max(plot_points_x) if plot_points_x else 1
    y_min_data = min(plot_points_y) if plot_points_y else -1
    y_max_data = max(plot_points_y) if plot_points_y else 1

    margin_x = max(2.0, abs(x_max_data - x_min_data) * 0.5)
    margin_y = max(2.0, abs(y_max_data - y_min_data) * 0.5)

    calc_x_min, calc_x_max = x_min_data - margin_x, x_max_data + margin_x
    calc_y_min, calc_y_max = y_min_data - margin_y, y_max_data + margin_y

    small_axis_offset = 0.1
    if variable_types[0] == '>=0' and calc_x_min < -small_axis_offset * margin_x : calc_x_min = -small_axis_offset * margin_x
    elif variable_types[0] == '<=0' and calc_x_max > small_axis_offset * margin_x : calc_x_max = small_axis_offset * margin_x

    if variable_types[1] == '>=0' and calc_y_min < -small_axis_offset * margin_y : calc_y_min = -small_axis_offset * margin_y
    elif variable_types[1] == '<=0' and calc_y_max > small_axis_offset * margin_y : calc_y_max = small_axis_offset * margin_y

    if abs(calc_x_max - calc_x_min) < 1.0: calc_x_min -=0.5; calc_x_max +=0.5
    if abs(calc_y_max - calc_y_min) < 1.0: calc_y_min -=0.5; calc_y_max +=0.5

    grid_points = 250
    x_grid = np.linspace(calc_x_min, calc_x_max, grid_points)
    y_grid = np.linspace(calc_y_min, calc_y_max, grid_points)
    X_mesh, Y_mesh = np.meshgrid(x_grid, y_grid)

    feasible_mask = np.ones(X_mesh.shape, dtype=bool)
    for i in range(len(A_orig)):
        val = A_orig[i][0] * X_mesh + A_orig[i][1] * Y_mesh
        if constraint_types_orig[i] == '<=': feasible_mask &= (val <= b_orig[i] + APP_TOLERANCE)
        elif constraint_types_orig[i] == '>=': feasible_mask &= (val >= b_orig[i] - APP_TOLERANCE)
        elif constraint_types_orig[i] == '=': feasible_mask &= (np.abs(val - b_orig[i]) < 0.01) # Dung sai cho ràng buộc bằng

    if variable_types[0] == '>=0': feasible_mask &= (X_mesh >= -APP_TOLERANCE)
    elif variable_types[0] == '<=0': feasible_mask &= (X_mesh <= APP_TOLERANCE)
    if variable_types[1] == '>=0': feasible_mask &= (Y_mesh >= -APP_TOLERANCE)
    elif variable_types[1] == '<=0': feasible_mask &= (Y_mesh <= APP_TOLERANCE)

    ax.imshow(feasible_mask.astype(int), extent=(calc_x_min, calc_x_max, calc_y_min, calc_y_max),
              origin='lower', cmap="Greens", alpha=0.3, aspect='auto')

    line_plot_x_vals = np.array([calc_x_min, calc_x_max])
    for i in range(len(A_orig)):
        a1, a2 = A_orig[i][0], A_orig[i][1]
        bi = b_orig[i]
        label_terms = []
        if abs(a1) > APP_TOLERANCE: label_terms.append(f"{a1:g}x₁")
        if abs(a2) > APP_TOLERANCE:
            op_str = " + " if a2 > 0 and label_terms else (" - " if a2 < 0 and label_terms else "")
            val_str = f"{abs(a2):g}" if abs(abs(a2)-1.0) > APP_TOLERANCE or not label_terms else ""
            if not label_terms and a2 < 0 : op_str = "-"
            label_terms.append(f"{op_str}{val_str}x₂" if label_terms else f"{a2:g}x₂")
        if not label_terms: label_terms.append("0")
        op_map = {'<=': '≤', '>=': '≥', '=': '='}
        label = f"{''.join(label_terms)} {op_map[constraint_types_orig[i]]} {bi:g}"
        if abs(a2) > APP_TOLERANCE:
            y_line_vals = (bi - a1 * line_plot_x_vals) / a2
            ax.plot(line_plot_x_vals, y_line_vals, label=label, lw=1.5)
        elif abs(a1) > APP_TOLERANCE:
            ax.axvline(x=bi / a1, label=label, lw=1.5)

    if variable_types[0] == '>=0': ax.axvline(x=0, color='gray', linestyle='--', lw=1, label='x₁ ≥ 0')
    elif variable_types[0] == '<=0': ax.axvline(x=0, color='gray', linestyle='--', lw=1, label='x₁ ≤ 0')
    if variable_types[1] == '>=0': ax.axhline(y=0, color='gray', linestyle='--', lw=1, label='x₂ ≥ 0')
    elif variable_types[1] == '<=0': ax.axhline(y=0, color='gray', linestyle='--', lw=1, label='x₂ ≤ 0')

    if has_valid_solution_for_plot:
        ax.plot(x_opt, y_opt, 'o', color='red', markersize=10, markeredgecolor='black',
                label=f'Tối ưu: ({x_opt:.2f}, {y_opt:.2f})', zorder=5)

    plot_title = "Biểu đồ miền khả thi"
    if has_valid_solution_for_plot: plot_title += " và điểm tối ưu"
    elif status_str.lower().startswith('vô nghiệm'): plot_title += " (Vô nghiệm)"
    elif status_str.lower().startswith('không giới nội'): plot_title += " (Không giới nội)"

    ax.set_title(plot_title, fontsize=14)
    ax.set_xlabel("x₁", fontsize=12)
    ax.set_ylabel("x₂", fontsize=12)
    ax.grid(True, linestyle=':', alpha=0.7)

    handles, labels_legend = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels_legend, fontsize=9, loc='center left', bbox_to_anchor=(1.01, 0.5))

    ax.set_xlim(calc_x_min, calc_x_max)
    ax.set_ylim(calc_y_min, calc_y_max)

    fig.tight_layout(rect=[0, 0, 0.82, 1])

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    plot_data_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(fig)

    return plot_data_base64


if __name__ == '__main__':
    if not os.path.exists('templates'):
        os.makedirs('templates')
    # Các file template nên được tạo riêng biệt, không nên viết code HTML cứng trong Python
    # if not os.path.exists('templates/index.html'):
    #     ...
    # if not os.path.exists('templates/result.html'):
    #     ...

    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=os.environ.get('FLASK_DEBUG', 'True').lower() == 'true')
