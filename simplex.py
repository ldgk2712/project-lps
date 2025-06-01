"""simplex_patch.py – Fix objective-type handling (MAX/MIN)

* Convert **MAX → MIN** by negating c when objective_type == 'max'.
* Keep c as‑is for MIN.
* Flip sign of z* back to MAX when returning.
* Retains preprocessing of ≥ constraints and step-by-step printing.
* Handles x_i <= 0 by substituting y_i = -x_i and using y_i in tableau.
* Updated tableau printing: z first, constants first in expressions.
"""

# ----------------------------- helpers --------------------------------------
from sympy import symbols, simplify, solve, Symbol, S
from typing import List, Dict, Tuple, Any, Union

def format_expression_for_printing(expression):
    """Định dạng biểu thức SymPy để hằng số đứng đầu và các biến được sắp xếp."""
    # Đơn giản hóa biểu thức
    expression = simplify(expression)
    
    # Tìm tất cả các biến trong biểu thức
    vars_in_expr = expression.free_symbols
    
    # Tách hệ số tự do và các phần chứa biến
    if vars_in_expr:
        const_term, var_terms = expression.as_coeff_add(*vars_in_expr)
    else:
        const_term = expression
        var_terms = []
    
    # Định dạng hệ số tự do: luôn hiển thị, kể cả khi bằng 0
    const_str = f"{float(const_term):.2f}" if const_term != 0 else "0.00"
    
    # Nếu không có phần chứa biến, trả về ngay hệ số tự do
    if not var_terms:
        return const_str
    
    # Xử lý các phần chứa biến
    var_term_dict = {}
    for term in var_terms:
        coeff, var = term.as_coeff_Mul()
        var_str = str(var) if var != 1 else ""
        if var_str:
            var_term_dict[var_str] = coeff
    
    # Sắp xếp các biến theo thứ tự ưu tiên (x, y, A, w, ...)
    sorted_vars = sorted(var_term_dict.keys(), key=lambda x: (
        0 if x.startswith('x') else 
        1 if x.startswith('y') else 
        2 if x.startswith('A') else 
        3 if x.startswith('w') else 4, x
    ))
    
    # Định dạng các phần chứa biến
    var_str_list = []
    for var in sorted_vars:
        coeff = var_term_dict[var]
        if coeff == 1:
            var_str_list.append(f"{var}")
        elif coeff == -1:
            var_str_list.append(f"-{var}")
        else:
            sign = "" if coeff > 0 else "-"
            var_str_list.append(f"{sign}{abs(float(coeff)):.2f}*{var}")
    
    # Kết hợp các phần chứa biến
    var_part = " + ".join(var_str_list).replace("+ -", "- ")
    
    # Trả về kết quả với hệ số tự do đứng đầu
    return f"{const_str} + {var_part}" if var_part else const_str

def _print_tableau(title: str, exprs: Dict[str, Any], basic_vars: List[str] = None, non_basic_vars: List[str] = None) -> None:
    """
    In bảng đơn hình với định dạng cải tiến:
    1. Dòng 'z' được in đầu tiên.
    2. Hằng số luôn đứng đầu trong biểu thức.
    3. Hiển thị biến cơ bản và không cơ bản (nếu có).
    """
    print(f"\n{'=' * 60}")
    print(f"{title}")
    print(f"{'-' * 60}")

    # Sắp xếp dòng: z đầu tiên
    ordered_var_names = ['z'] if 'z' in exprs else []
    decision_artificial_vars = sorted([
        k for k in exprs if k != 'z' and (k.startswith('x') or k.startswith('y') or k.startswith('A'))
    ])
    slack_surplus_vars = sorted([
        k for k in exprs if k != 'z' and k.startswith('w')
    ])
    ordered_var_names.extend(decision_artificial_vars + slack_surplus_vars)

    # In bảng
    print(f"{'Biến':<10} | {'Biểu thức':<50}")
    print(f"{'-' * 10} | {'-' * 50}")
    for var_name in ordered_var_names:
        if var_name in exprs:
            formatted_str = format_expression_for_printing(exprs[var_name])
            print(f"{var_name:<10} | {formatted_str:<50}")

    if basic_vars or non_basic_vars:
        print(f"{'-' * 60}")
        if basic_vars:
            print(f"Biến cơ bản: {', '.join(basic_vars)}")
        if non_basic_vars:
            print(f"Biến không cơ bản: {', '.join(non_basic_vars)}")
    print(f"{'=' * 60}")

def _simplex_min(A: List[List[float]], b: List[float], c: List[float], constraint_types: List[str], variable_types: List[str], objective_type: str) -> Tuple[str, Union[float, None], Dict[str, float], Dict[str, Any]]:
    m, n_original = len(A), len(A[0])  # m: số ràng buộc, n_original: số biến gốc

    # Kiểm tra đầu vào
    if len(b) != m or len(constraint_types) != m or len(c) != n_original or len(variable_types) != n_original:
        raise ValueError("Kích thước đầu vào không khớp")
    for i, ct in enumerate(constraint_types):
        if ct.strip() not in ['<=', '>=', '=']:
            raise ValueError(f"Loại ràng buộc không hợp lệ: {ct}")
        if ct == '<=' and b[i] < 0:
            raise ValueError(f"Ràng buộc <= không thể có b[{i}] âm: {b[i]}")

    A_standard = [row[:] for row in A]
    b_standard = b[:]
    c_standard = c[:]

    # 1. Xử lý biến x_i <= 0
    original_var_info_map = []
    for i in range(n_original):
        original_name = f"x{i+1}"
        is_transformed = (variable_types[i].strip() == '<=0')
        tableau_name = f"y{i+1}" if is_transformed else original_name
        if is_transformed:
            for j_row in range(m):
                A_standard[j_row][i] = -A_standard[j_row][i]
            c_standard[i] = -c_standard[i]
        original_var_info_map.append({
            'original_name': original_name,
            'tableau_name': tableau_name,
            'symbol': Symbol(tableau_name),
            'is_transformed': is_transformed,
            'original_idx': i
        })

    decision_symbols = [info['symbol'] for info in original_var_info_map]

    # 2. Xử lý ràng buộc và biến nhân tạo
    artificial_var_symbols = []
    num_artificial_vars_added = 0
    for constraint_idx in range(m):
        if constraint_types[constraint_idx].strip() in ['=', '>=']:
            tableau_col_idx_for_art_var = n_original + num_artificial_vars_added
            for r_idx in range(m):
                A_standard[r_idx].append(0)
            A_standard[constraint_idx][tableau_col_idx_for_art_var] = 1
            c_standard.append(1e10)  # Big-M
            art_var_name = f"A{num_artificial_vars_added + 1}"
            artificial_var_symbols.append(Symbol(art_var_name))
            num_artificial_vars_added += 1

    num_vars_in_tableau = n_original + num_artificial_vars_added
    all_tableau_symbols = decision_symbols + artificial_var_symbols

    # 3. Tạo biến bù (slack/surplus)
    w_symbols = list(symbols(f"w1:{m+1}"))

    # 4. Khởi tạo bảng đơn hình
    step = 0
    steps: Dict[str, Dict[str, Any]] = {}
    cur: Dict[str, Any] = {}
    basic_var_names = []
    non_basic_var_names = [str(s) for s in all_tableau_symbols]

    # Tạo hàm mục tiêu z trước
    z_expr = sum(float(c_standard[j_col]) * all_tableau_symbols[j_col] for j_col in range(num_vars_in_tableau))
    for i in range(m):
        if constraint_types[i].strip() in ['=', '>='] and artificial_var_symbols:
            art_var = artificial_var_symbols[-1] if constraint_types[i].strip() == '=' else artificial_var_symbols[i]
            z_expr += 1e10 * art_var
    cur['z'] = simplify(z_expr)  # Đảm bảo z được chuẩn hóa

    # Tạo các ràng buộc
    for i in range(m):
        ct = constraint_types[i].strip()
        expr = sum(float(A_standard[i][j_col]) * all_tableau_symbols[j_col] for j_col in range(num_vars_in_tableau))
        if ct == '>=':
            expr = -float(b_standard[i]) + expr
        else:  # <= or =
            expr = float(b_standard[i]) - expr
        cur[str(w_symbols[i])] = simplify(expr)
        if ct in ['<=', '='] or (ct == '>=' and artificial_var_symbols):
            basic_var_names.append(str(w_symbols[i]))
        else:
            non_basic_var_names.append(str(w_symbols[i]))

    steps['Step 0'] = cur.copy()
    _print_tableau('Step 0 (Khởi tạo - Initial Tableau)', cur, basic_var_names, non_basic_var_names)

    # 5. Vòng lặp Simplex
    while True:
        z_expr = cur['z']
        entering_var_name, most_neg_coeff = None, S.Zero

        for v_name_str in non_basic_var_names:
            v_sym = Symbol(v_name_str)
            coeff = z_expr.coeff(v_sym)
            if coeff < most_neg_coeff:
                most_neg_coeff = coeff
                entering_var_name = v_name_str

        if entering_var_name is None:
            if any(v.startswith('A') for v in basic_var_names):
                status = 'Infeasible'
                break
            status = 'Optimal'
            break

        leaving_var_name, min_ratio = None, float('inf')
        entering_var_symbol = Symbol(entering_var_name)
        no_positive_pivot_element_found = True

        for w_name_str in basic_var_names:
            expr_w = cur[w_name_str]
            coeff_entering_in_w_expr = expr_w.coeff(entering_var_symbol)
            if coeff_entering_in_w_expr < 0:
                no_positive_pivot_element_found = False
                subs_for_const = [(Symbol(nb_name), 0) for nb_name in non_basic_var_names]
                const_in_w_expr = simplify(expr_w.subs(subs_for_const))
                if not const_in_w_expr.is_number:
                    print(f"Lỗi: Hằng số không phải số: {const_in_w_expr} cho {w_name_str}")
                    continue
                const_val = float(const_in_w_expr)
                if const_val >= 0:
                    ratio = const_val / (-coeff_entering_in_w_expr)
                    if ratio < min_ratio:
                        min_ratio = ratio
                        leaving_var_name = w_name_str

        if no_positive_pivot_element_found:
            status = 'Unbounded'
            title = f"Step {step+1} (Biến vào {entering_var_name}, không có biến ra — Không giới nội)"
            steps[title] = cur.copy()
            _print_tableau(title, steps[title], basic_var_names, non_basic_var_names)
            break

        if leaving_var_name is None:
            status = 'Infeasible'
            title = f"Step {step+1} (Biến vào {entering_var_name}, không có biến ra hợp lệ — Vô nghiệm)"
            steps[title] = cur.copy()
            _print_tableau(title, steps[title], basic_var_names, non_basic_var_names)
            break

        step += 1
        pivot_eq_lhs = Symbol(leaving_var_name)
        pivot_eq_rhs = cur[leaving_var_name]
        solved_entering_expr_list = solve(pivot_eq_lhs - pivot_eq_rhs, entering_var_symbol)
        if not solved_entering_expr_list:
            print(f"Lỗi: Không giải được pivot cho {entering_var_name}. PT: {pivot_eq_lhs} = {pivot_eq_rhs}")
            status = "Error"
            break
        pivot_expr_for_entering = solved_entering_expr_list[0]

        new_cur = {entering_var_name: simplify(pivot_expr_for_entering.subs(Symbol(leaving_var_name), 0))}
        for var_name, old_expr in cur.items():
            if var_name == leaving_var_name:
                continue
            new_expr = simplify(old_expr.subs(entering_var_symbol, pivot_expr_for_entering))
            new_cur[var_name] = new_expr

        basic_var_names.remove(leaving_var_name)
        basic_var_names.append(entering_var_name)
        non_basic_var_names.remove(entering_var_name)
        non_basic_var_names.append(leaving_var_name)

        cur = new_cur
        title = f"Step {step} (Biến vào: {entering_var_name}, Biến ra: {leaving_var_name})"
        steps[title] = cur.copy()
        _print_tableau(title, cur, basic_var_names, non_basic_var_names)

    # 6. Trích xuất kết quả
    z_star_final_value = None
    solution_final_values = {}

    if status == 'Optimal':
        subs_non_basic_to_zero = {Symbol(nb_name): 0 for nb_name in non_basic_var_names}
        opt_vals_tableau = {}
        all_vars_in_final_tableau = basic_var_names + non_basic_var_names
        for v_name_str in all_vars_in_final_tableau:
            opt_vals_tableau[v_name_str] = float(cur[v_name_str].subs(subs_non_basic_to_zero)) if v_name_str in basic_var_names else 0.0

        for info in original_var_info_map:
            original_x_name = info['original_name']
            tableau_var_name = info['tableau_name']
            val_in_tableau = opt_vals_tableau.get(tableau_var_name, 0.0)
            solution_final_values[original_x_name] = -val_in_tableau if info['is_transformed'] else val_in_tableau

        current_z_star = sum(float(c[i]) * solution_final_values.get(original_var_info_map[i]['original_name'], 0.0) for i in range(n_original))
        z_star_final_value = current_z_star

    elif status == 'Unbounded':
        z_star_final_value = float('-inf')
    elif status == 'Infeasible':
        z_star_final_value = float('inf')

    # Sau khi tính toán solution_final_values
    for key in solution_final_values:
        if abs(solution_final_values[key]) < 1e-10:  # Kiểm tra dung sai
            solution_final_values[key] = 0.0         # Gán về 0 nếu giá trị rất gần 0

    # Sau khi tính toán z_star_final_value
    if status == 'Optimal' and abs(z_star_final_value) < 1e-10:
        z_star_final_value = 0.0  # Đảm bảo giá trị tối ưu cũng được xử lý
    
    return status, z_star_final_value, solution_final_values, steps

# --------------------------- public wrapper ---------------------------------
def auto_simplex(
    A: List[List[float]],
    b: List[float],
    c: List[float], 
    constraint_types: List[str],
    objective_type: str = 'max',
    variable_types: List[str] | None = None,
) -> Dict[str, Any]:
    """
    Hàm chính để giải bài toán lập trình tuyến tính bằng thuật toán Simplex.
    Chuyển đổi bài toán MAX thành MIN nếu cần và trả về kết quả với định dạng chuẩn.

    Args:
        A: Ma trận hệ số của các ràng buộc (m x n).
        b: Vector hằng số của các ràng buộc (m).
        c: Vector hệ số của hàm mục tiêu (n).
        constraint_types: Loại ràng buộc ('<=', '>=', '=') cho từng ràng buộc.
        objective_type: Loại bài toán ('max' hoặc 'min'), mặc định là 'max'.
        variable_types: Loại biến ('>=0' hoặc '<=0'), mặc định là '>=0'.

    Returns:
        Dictionary chứa:
            - status: Trạng thái giải ('Tối ưu', 'Không giới nội', 'Vô nghiệm', 'Lỗi').
            - z: Giá trị hàm mục tiêu tại nghiệm tối ưu (hoặc None nếu vô nghiệm).
            - solution: Dictionary chứa giá trị các biến tại nghiệm tối ưu.
            - steps: Dictionary chứa các bước của bảng đơn hình.
    """
    # 1. Kiểm tra đầu vào
    num_constraints, num_vars = len(A), len(c)
    if not A or not all(len(row) == num_vars for row in A):
        raise ValueError("Ma trận A không hợp lệ hoặc không khớp với số biến")
    if len(b) != num_constraints:
        raise ValueError("Vector b phải có độ dài bằng số ràng buộc")
    if len(constraint_types) != num_constraints:
        raise ValueError("Số loại ràng buộc phải bằng số ràng buộc")
    if objective_type.strip().lower() not in ['max', 'min']:
        raise ValueError("objective_type phải là 'max' hoặc 'min'")
    if variable_types is None:
        variable_types = ['>=0'] * num_vars
    elif len(variable_types) != num_vars:
        raise ValueError("Độ dài variable_types phải bằng số biến")
    for vt in variable_types:
        if vt.strip() not in ['>=0', '<=0']:
            raise ValueError(f"Loại biến không hợp lệ: {vt}")
    for ct in constraint_types:
        if ct.strip() not in ['<=', '>=', '=']:
            raise ValueError(f"Loại ràng buộc không hợp lệ: {ct}")

    # 2. Chuẩn bị dữ liệu cho bài toán MIN
    c_orig = c[:]
    flip_z_sign_back = objective_type.strip().lower().startswith('max')
    c_eff_for_min_problem = [-ci for ci in c_orig] if flip_z_sign_back else c_orig[:]

    # 3. Gọi hàm _simplex_min
    try:
        status, z_star_from_min_problem, opt_vals_x_final, steps = _simplex_min(
            A, b, c_eff_for_min_problem, constraint_types, variable_types, objective_type
        )
    except Exception as e:
        return {
            'status': 'Lỗi (Error)',
            'z': None,
            'solution': {},
            'steps': {},
            'error_message': str(e)
        }

    # 4. Xử lý giá trị z cuối cùng
    final_z_value = z_star_from_min_problem
    if status == 'Optimal':
        if flip_z_sign_back and z_star_from_min_problem is not None:
            final_z_value = -z_star_from_min_problem
    elif status == 'Unbounded':
        final_z_value = float('inf') if flip_z_sign_back else float('-inf')
    elif status == 'Infeasible':
        final_z_value = None

    # 5. Chuẩn bị nghiệm để trả về
    solution_dict_to_return = opt_vals_x_final if status == 'Optimal' else {}

    # 6. Ánh xạ trạng thái sang tiếng Việt
    status_map = {
        'Optimal': 'Tối ưu (Optimal)',
        'Unbounded': 'Không giới nội (Unbounded)',
        'Infeasible': 'Vô nghiệm (Infeasible)',
        'Error': 'Lỗi (Error)'
    }

    # Sau khi thuật toán hoàn tất và trước khi return
    # Định dạng các bước với hệ số tự do đứng đầu
    formatted_steps = {}
    for step_title, tableau in steps.items():
        formatted_tableau = {}
        for var, expr in tableau.items():
            formatted_tableau[var] = format_expression_for_printing(expr)
        formatted_steps[step_title] = formatted_tableau

    # Trả về kết quả với steps đã định dạng
    return {
        'status': status_map.get(status, status),
        'z': final_z_value,
        'solution': solution_dict_to_return,
        'steps': formatted_steps,
    }

if __name__ == '__main__':
    # Ví dụ kiểm tra với biến x_i <= 0
    print("Ví dụ 1: MAX Z = x1")
    print("s.t. x1 <= -5")
    print("     x1 >= -10 (tức là -x1 <= 10)")
    
    A1_single_var = [[1], [-1]]
    b1_single_var = [-5, 10]
    c1_single_var = [1]
    constraint_types1_single_var = ['<=', '<='] 
    variable_types1_single_var = ['<=0'] 
    objective_type1_single_var = 'max'

    result1 = auto_simplex(A1_single_var, b1_single_var, c1_single_var, constraint_types1_single_var, objective_type1_single_var, variable_types1_single_var)
    print("\nKết quả Ví dụ 1 (x1 <= -5, x1 >= -10, MAX Z = x1):")
    print(f"Status: {result1['status']}")
    print(f"z = {result1['z']}") 
    print(f"Solution: {result1['solution']}") 
    # In các bước để xem định dạng mới:
    # for title_step, table_data in result1['steps'].items():
    #    _print_tableau(title_step, table_data)


    print("\nVí dụ 2: MIN Z = 2x1 + 3x2, x1 <= 0, x2 >=0 (mặc định)")
    print("s.t.  x1 +  x2 >= 1")
    print("     2x1 + 5x2 >= 10")
    A2 = [[1,1], [2,5]]
    b2 = [1, 10]
    c2_orig = [2,3] 
    constraint_types2 = ['>=', '>=']
    variable_types2 = ['<=0', '>=0'] 
    objective_type2 = 'min'
    result2 = auto_simplex(A2, b2, c2_orig, constraint_types2, objective_type2, variable_types2)
    print("\nKết quả Ví dụ 2:")
    print(f"Status: {result2['status']}")
    print(f"z = {result2['z']}")
    print(f"Solution: {result2['solution']}")
    # for title_step, table_data in result2['steps'].items():
    #    _print_tableau(title_step, table_data)


    print("\nVí dụ 3: MAX Z = x1 + x2, x1 <= -1, x2 <= -1")
    A3 = [[1,0], [0,1]] 
    b3 = [-1, -1] 
    c3_orig = [1,1]   
    constraint_types3 = ['<=', '<=']
    variable_types3 = ['<=0', '<=0'] 
    objective_type3 = 'max'
    result3 = auto_simplex(A3, b3, c3_orig, constraint_types3, objective_type3, variable_types3)
    print("\nKết quả Ví dụ 3:")
    print(f"Status: {result3['status']}")
    print(f"z = {result3['z']}") 
    print(f"Solution: {result3['solution']}") 
    # for title_step, table_data in result3['steps'].items():
    #    _print_tableau(title_step, table_data)

